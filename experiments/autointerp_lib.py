"""autointerp_lib.py — lightweight, OpenAI-backed automated interpretability.

Intentionally small: it labels and scores only a *few* SAE features (not all 65k)
so the steering notebook can pick interpretable features *data-drivenly with an LLM*
instead of by keyword/purity heuristics, and then auto-write a steering seed prompt
for each. Three LLM-backed steps, all cheap:

  1. explain_feature   — describe what a feature detects, from its top-activating
                         examples (strong tokens wrapped in <<...>>).
  2. detection_score   — a light "autointerp score": give the explanation + a mix of
                         activating and non-activating snippets and measure how well
                         the LLM separates them (balanced accuracy). High ⇒ the
                         description really predicts when the feature fires.
  3. make_seed_prompt  — turn the description into a short, neutral steering prompt
                         (+ target tokens) for the causal steering experiments.

A heuristic prefilter (rank_candidate_features) trims the 65k features to a handful
of plausibly-clean candidates BEFORE any LLM call, so the whole pass is a few dozen
cheap requests per SAE.

Setup: put ``OPENAI_API_KEY=sk-...`` in the repo ``.env`` (or the environment).
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# OpenAI client
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    """Populate os.environ from the repo .env (so OPENAI_API_KEY can live there)."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client(api_key: str | None = None, base_url: str | None = None,
               verify_ssl: bool | None = None):
    """An OpenAI(-compatible) client, reading config from args / env / .env.

    Supports OpenAI-compatible proxies (e.g. AMD primus-safe) via OPENAI_BASE_URL.
    Such internal proxies often present certs httpx won't verify, so when a base_url
    is set we default to an unverified httpx client (override with OPENAI_VERIFY_SSL=1
    or verify_ssl=True). With a base_url, use the proxy's exact model ids from
    ``client.models.list()`` (e.g. ``gpt-5.4-mini`` — bare, not ``openai/``-prefixed).
    """
    from openai import OpenAI

    _load_env()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenAI API key. Add OPENAI_API_KEY=... to efficient_sae/.env "
            "or pass api_key=...")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
        if verify_ssl is None:
            verify_ssl = os.environ.get("OPENAI_VERIFY_SSL", "0").lower() in (
                "1", "true", "yes")
        if not verify_ssl:
            import httpx
            kwargs["http_client"] = httpx.Client(verify=False)
    return OpenAI(**kwargs)


def _chat(client, model, messages, temperature=0.0, max_tokens=300):
    """One chat completion -> stripped text.

    Tolerant to model/proxy parameter differences: newer models (GPT-5 family) reject
    ``max_tokens`` (want ``max_completion_tokens``) and only allow the default
    ``temperature``, while older ones want ``max_tokens``. We try the variants in turn.
    """
    base = {"model": model, "messages": messages}
    attempts = [
        {**base, "temperature": temperature, "max_tokens": max_tokens},
        {**base, "temperature": temperature, "max_completion_tokens": max_tokens},
        {**base, "max_completion_tokens": max_tokens},
        base,
    ]
    last_err = None
    for kw in attempts:
        try:
            resp = client.chat.completions.create(**kw)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 — proxy/model param differences
            last_err = e
    raise last_err


def _extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (handles ```json fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
    return json.loads(text)  # last resort: raise


# --------------------------------------------------------------------------- #
# Candidate prefilter (no LLM) — trim 65k features to a clean handful
# --------------------------------------------------------------------------- #
_MARKUP_HINTS = ("http", "www", "{", "}", "\\", "/>", "</", "===", "```", "usepackage",
                 "\\n", "\\t")


def _looks_like_content(tok_str: str) -> bool:
    """True if a token looks like a real content word (not punctuation/markup/number).

    This is what separates *interesting* concept features from the structural features
    (newlines, punctuation, URLs, code, citations, boilerplate) that dominate max-act
    rankings in web/code-trained SAEs.
    """
    s = (tok_str or "").strip()
    if len(s) < 3 or not any(c.isalpha() for c in s):
        return False
    # mostly alphabetic (allow an internal hyphen/apostrophe), no markup substrings
    alpha = sum(c.isalpha() for c in s)
    if alpha / len(s) < 0.8:
        return False
    low = s.lower()
    return not any(h in low for h in _MARKUP_HINTS)


def _dominant_token(tokenizer, top_tok_ids):
    """Most frequent token among a feature's top activations, decoded to a string."""
    ids = [int(t) for t in top_tok_ids if int(t) >= 0]
    if not ids:
        return ""
    vals, counts = np.unique(np.array(ids), return_counts=True)
    tid = int(vals[int(counts.argmax())])
    try:
        return tokenizer.decode([tid])
    except Exception:
        return ""


def rank_candidate_features(scan_dir, n=24, min_max_act=8.0, min_freq=2e-5,
                            max_freq=2e-2, min_purity=0.0, tokenizer=None,
                            content_only=False, working_multiple=12):
    """Pick ~``n`` candidate features worth labelling, from the feature scan .npz.

    Keeps strongly-activating features in a specificity band, ranked by max activation.
    When ``content_only`` is set (and a ``tokenizer`` is given), features whose dominant
    top-activating token is punctuation / whitespace / digits / markup are skipped — a
    cheap, no-LLM way to surface *interesting* semantic features instead of structural
    ones. Returns dicts: feature, max_act, firing_freq, top_token_share, top_token.
    """
    scan_dir = Path(scan_dir)
    npz_path = scan_dir / "concrete_features.npz"
    if not npz_path.exists():                      # allow passing the SAE dir itself
        npz_path = scan_dir / "steering" / "concrete_features.npz"
    d = np.load(npz_path)
    max_act = d["max_activation"].astype("float32")
    freq = d["firing_freq"].astype("float32")
    share = d["top_token_share"].astype("float32")
    top_toks = d["top_toks"] if "top_toks" in d else None
    ok = ((max_act >= min_max_act) & (freq >= min_freq)
          & (freq <= max_freq) & (share >= min_purity) & np.isfinite(max_act))
    order = np.where(ok)[0]
    order = order[np.argsort(-max_act[order])]
    gate = content_only and tokenizer is not None and top_toks is not None
    if gate:
        order = order[: n * working_multiple]      # only decode the strongest working set

    out = []
    for i in order:
        tok = (_dominant_token(tokenizer, top_toks[i])
               if (tokenizer is not None and top_toks is not None) else "")
        if gate and not _looks_like_content(tok):
            continue
        out.append({"feature": int(i), "max_act": float(max_act[i]),
                    "firing_freq": float(freq[i]),
                    "top_token_share": float(share[i]), "top_token": tok})
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Rendering top-activating examples for the explainer
# --------------------------------------------------------------------------- #
def _clean(tok: str) -> str:
    return tok.replace("\n", "\\n").replace("\t", " ")


def render_example(str_tokens, acts, thresh_frac=0.5, max_chars=320):
    """One example as plain text with the strongly-firing tokens wrapped in <<...>>."""
    acts = list(acts)
    m = max(acts) if acts else 0.0
    m = m or 1.0
    out = []
    for tok, a in zip(str_tokens, acts):
        t = _clean(tok)
        out.append(f"<<{t}>>" if (a > 0 and a >= thresh_frac * m) else t)
    s = "".join(out).strip()
    return s[:max_chars]


def render_examples(examples, max_examples=10, **kw):
    """Numbered block of rendered examples for the explainer prompt."""
    lines = [render_example(e["str_tokens"], e["acts"], **kw)
             for e in examples[:max_examples]]
    return "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines) if ln)


def plain_snippets(examples, n=6, max_chars=240):
    """Plain (un-highlighted) activating snippets, for detection scoring positives."""
    out = []
    for e in examples[:n]:
        s = "".join(_clean(t) for t in e["str_tokens"]).strip()
        if s:
            out.append(s[:max_chars])
    return out


# --------------------------------------------------------------------------- #
# 1) Explain — what does this feature detect?
# --------------------------------------------------------------------------- #
_EXPLAIN_SYS = (
    "You are an interpretability researcher analyzing a single feature (neuron) of a "
    "sparse autoencoder trained on a language model's activations. You are shown text "
    "excerpts where the feature activates most strongly; the specific tokens it fires "
    "on are wrapped in <<double angle brackets>>. Identify what the feature detects. "
    "Reply with ONE concise noun phrase of at most 8 words (no preamble, no quotes, no "
    "trailing period). Prefer a concrete, specific concept over a vague one.")


def explain_feature(client, examples, model="gpt-4o-mini", max_examples=10):
    """Short natural-language description of the feature from its examples."""
    block = render_examples(examples, max_examples=max_examples)
    if not block:
        return ""
    desc = _chat(client, model, [
        {"role": "system", "content": _EXPLAIN_SYS},
        {"role": "user", "content": f"Excerpts:\n{block}\n\nDescription:"}],
        temperature=0.0, max_tokens=40)
    return desc.strip().strip('"').rstrip(".")


_EXPLAIN_RATE_SYS = (
    "You are an interpretability researcher analyzing one feature of a sparse "
    "autoencoder over a language model. You see excerpts where it fires hardest, with "
    "the firing tokens wrapped in <<double angle brackets>>. Return ONLY a JSON object:\n"
    '  "description": a concise noun phrase (<=8 words) for what it detects;\n'
    '  "category": one of "semantic_concept", "named_entity", "topic", "syntax", '
    '"formatting", "boilerplate", "code", "other";\n'
    '  "interest": a float 0..1 for how INTERESTING and steerable the feature is for a '
    "demo. Use this scale:\n"
    "    0.8-1.0 = a distinctive real-world concept, entity, topic, tone, emotion, or "
    "behaviour (e.g. volcanoes, the Golden Gate Bridge, legal language, anger, deception);\n"
    "    0.5-0.7 = a recognizable but broad theme (e.g. sports, time, animals);\n"
    "    0.2-0.4 = a generic content word with little theme;\n"
    "    0.0-0.1 = punctuation, whitespace, markup, code syntax, citation/license "
    "boilerplate, or generic function words.\n"
    "Judge by the CONCEPT, not by how many examples you see.")


def explain_and_rate(client, examples, model="gpt-4o-mini", max_examples=10):
    """Describe a feature AND rate how interesting it is, in one call.

    Returns {description, category, interest}. Falls back to a plain description
    (interest=nan) if the model doesn't return parseable JSON.
    """
    block = render_examples(examples, max_examples=max_examples)
    if not block:
        return {"description": "", "category": "other", "interest": float("nan")}
    raw = _chat(client, model, [
        {"role": "system", "content": _EXPLAIN_RATE_SYS},
        {"role": "user", "content": f"Excerpts:\n{block}\n\nJSON:"}],
        temperature=0.0, max_tokens=120)
    try:
        obj = _extract_json(raw)
        return {
            "description": str(obj.get("description", "")).strip().strip('"').rstrip("."),
            "category": str(obj.get("category", "other")).strip(),
            "interest": float(obj.get("interest", float("nan"))),
        }
    except Exception:
        return {"description": raw.strip().strip('"').rstrip("."),
                "category": "other", "interest": float("nan")}


# --------------------------------------------------------------------------- #
# 2) Detection score — a light autointerp score (balanced accuracy)
# --------------------------------------------------------------------------- #
_DETECT_SYS = (
    "You judge whether a described language-model feature would activate on a text. "
    "You are given a feature DESCRIPTION and a numbered list of texts. For each text, "
    "output 1 if the feature (as described) would clearly fire on it, else 0. "
    "Respond with ONLY a JSON array of 0/1 integers, one per text, in order.")


def detection_score(client, description, pos_snippets, neg_snippets,
                    model="gpt-4o-mini", seed=0):
    """Balanced accuracy of the LLM separating activating vs non-activating snippets.

    This is the "autointerp score": a description that truly captures the feature lets
    the LLM tell activating texts from random ones. Returns (score, n_items, detail).
    """
    items = ([{"label": 1, "text": s} for s in pos_snippets]
             + [{"label": 0, "text": s} for s in neg_snippets])
    if not items or not pos_snippets or not neg_snippets:
        return float("nan"), len(items), {}
    rng = random.Random(seed)
    rng.shuffle(items)
    numbered = "\n".join(f"{i + 1}. {it['text']}" for i, it in enumerate(items))
    raw = _chat(client, model, [
        {"role": "system", "content": _DETECT_SYS},
        {"role": "user", "content":
            f'DESCRIPTION: "{description}"\n\nTexts:\n{numbered}\n\nJSON array:'}],
        temperature=0.0, max_tokens=10 + 3 * len(items))
    try:
        preds = _extract_json(raw)
        preds = [int(p) for p in preds][:len(items)]
    except Exception:
        return float("nan"), len(items), {"error": raw[:200]}
    if len(preds) != len(items):
        preds = (preds + [0] * len(items))[:len(items)]
    y = np.array([it["label"] for it in items])
    p = np.array(preds)
    # balanced accuracy = mean of per-class recall (robust to class imbalance)
    tpr = float((p[y == 1] == 1).mean()) if (y == 1).any() else 0.0
    tnr = float((p[y == 0] == 0).mean()) if (y == 0).any() else 0.0
    return 0.5 * (tpr + tnr), len(items), {"tpr": tpr, "tnr": tnr}


# --------------------------------------------------------------------------- #
# 3) Seed prompt — turn a description into a steering probe
# --------------------------------------------------------------------------- #
_PROMPT_SYS = (
    "You design probes for causal steering experiments on a single language-model "
    "feature. Given a description of what the feature detects, produce:\n"
    '  - "prompt": a short, fluent, NEUTRAL sentence opening (at most 12 words, no '
    "ending punctuation) that does NOT itself mention the concept, but is a natural "
    "lead-in where the concept could plausibly continue.\n"
    '  - "targets": 3 to 5 lowercase continuation tokens the concept would produce, '
    "each a single word starting with a leading space.\n"
    'Example for "the Golden Gate Bridge": '
    '{"prompt":"On our drive into the city we suddenly saw the",'
    '"targets":[" bridge"," Golden"," fog"," bay"]}\n'
    "Respond with ONLY a JSON object with keys prompt and targets.")


def to_single_token_targets(model, targets, max_targets=5):
    """Reduce each target string to its FIRST model token (as a string), keep only
    valid single tokens, de-duplicate.

    ``target_logprob`` / ``steering_metric`` score the next-token log-prob and require
    SINGLE-token targets, but LLM-proposed words (e.g. " acetylation") often tokenize
    into several pieces. The first piece is exactly the next token the model would emit
    for that word, so it is the right single-token proxy.
    """
    out, seen = [], set()
    for t in targets:
        if not t:
            continue
        try:
            pieces = model.to_str_tokens(t, prepend_bos=False)
        except Exception:
            continue
        if not pieces:
            continue
        s = pieces[0]
        if not s.strip() and len(pieces) >= 2:   # bare leading-space piece -> merge next
            s = s + pieces[1]
        if not s.strip() or s in seen:
            continue
        try:
            model.to_single_token(s)             # verify it really is one token
        except Exception:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_targets:
            break
    return out


def make_seed_prompt(client, description, model="gpt-4o-mini",
                     default_prompt="I went for a walk and"):
    """LLM-written neutral steering prompt + target tokens for a feature description."""
    raw = _chat(client, model, [
        {"role": "system", "content": _PROMPT_SYS},
        {"role": "user", "content": f'Feature description: "{description}"\n\nJSON:'}],
        temperature=0.3, max_tokens=120)
    try:
        obj = _extract_json(raw)
        prompt = str(obj.get("prompt") or default_prompt).strip().rstrip(".")
        targets = [str(t) for t in (obj.get("targets") or []) if str(t).strip()]
        targets = [t if t.startswith(" ") else " " + t.strip() for t in targets]
        return prompt, targets[:5]
    except Exception:
        return default_prompt, []


# --------------------------------------------------------------------------- #
# One-stop: explain + score + seed prompt for a single feature
# --------------------------------------------------------------------------- #
def autointerp_feature(client, examples, neg_snippets, *, explainer_model="gpt-4o-mini",
                       scorer_model="gpt-4o-mini", prompt_model="gpt-4o-mini",
                       n_pos=6, score=True, seed=0):
    """Full light autointerp for one feature.

    examples:     its top-activating examples ({str_tokens, acts, max_act}).
    neg_snippets: shared pool of non-activating texts (detection negatives).
    Returns {description, category, interest, score, n_scored, tpr, tnr, prompt, targets}.
    """
    rated = explain_and_rate(client, examples, model=explainer_model)
    description = rated["description"]
    out = {"description": description, "category": rated["category"],
           "interest": rated["interest"], "score": float("nan"),
           "n_scored": 0, "prompt": "", "targets": []}
    if not description:
        return out
    if score and neg_snippets:
        pos = plain_snippets(examples, n=n_pos)
        negs = list(neg_snippets)
        random.Random(seed).shuffle(negs)
        sc, n, detail = detection_score(
            client, description, pos, negs[:max(n_pos, 4)], model=scorer_model, seed=seed)
        out.update(score=sc, n_scored=n, tpr=detail.get("tpr"), tnr=detail.get("tnr"))
    prompt, targets = make_seed_prompt(client, description, model=prompt_model)
    out.update(prompt=prompt, targets=targets)
    return out
