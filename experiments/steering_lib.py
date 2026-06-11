"""steering_lib.py — reusable pieces for the SAE steering experiment.

Everything the steering analysis notebook needs, factored out so the notebook stays
thin and the logic is importable / testable:

  * load_model_and_sae   load a HookedTransformer + an exported inference SAE (the base
                         model is read from the SAE's own cfg, and cached so multiple
                         SAEs sharing a base model reuse one copy of the LLM).
  * feature_max_act      the feature's max activation observed over the scan dataset
                         (the "max value in the dataset" used to scale steering).
  * find_feature_by_keyword
                         locate "the same concept" feature independently in each SAE by
                         matching its top-activating-token class to a keyword.
  * per_token_acts       per-token activations of one feature on a piece of text.
  * token_heatmap_html   Anthropic-style (scaling-monosemanticity) colored-token heatmap.
  * steer_generate       generate with the feature clamped to (coeff x max_act) along its
                         decoder direction, for +x and -x sweeps, steering on/off.
  * target_logprob       Δ log-prob of concept tokens under steering (optional metric).

These are model-agnostic: the model + hook always come from the SAE config.
"""

from __future__ import annotations

import html
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch

_MODEL_CACHE: dict[tuple, object] = {}


def _load_env() -> None:
    """Load repo .env (gemma-2-2b is gated)."""
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def layer_from_hook(hook_name: str) -> int | None:
    import re
    m = re.search(r"blocks\.(\d+)\.", hook_name or "")
    return int(m.group(1)) if m else None


def _step_of(path) -> int:
    """Training step encoded in a checkpoint path ('final_499998720', '24414')."""
    import re
    for part in reversed(Path(path).parts):
        m = re.fullmatch(r"(?:final_)?(\d+)", part)
        if m:
            return int(m.group(1))
    return -1


def _autodetect_member(run_dir: Path) -> str:
    """The single SAE member in a run dir (a subdir with cfg.json that isn't a
    bookkeeping dir). Raises if there are zero or several, so callers pass --member."""
    skip = {"checkpoints", "saebench_eval", "steering", "wandb", "inference_saes",
            "_inference"}
    members = sorted(d.name for d in run_dir.iterdir()
                     if d.is_dir() and d.name not in skip and (d / "cfg.json").exists())
    if len(members) == 1:
        return members[0]
    raise ValueError(
        f"{run_dir} has {len(members)} members {members}; pass member= explicitly.")


def latest_checkpoint(run_dir, member: str | None = None):
    """Resolve the LATEST available checkpoint of ``member`` under ``run_dir``.

    Returns ``(load_dir, step, is_final)``. The fully-trained final export
    (``run_dir/member``) wins when present (it's the end of training); otherwise the
    highest intermediate training step under ``run_dir/checkpoints/<hash>/<step>/member``
    is used — so the notebook works on a run that is still training. Mirrors the
    checkpoint discovery in eval_saebench.py.
    """
    import glob as _glob
    run_dir = Path(run_dir)
    if member is None:
        member = _autodetect_member(run_dir)

    ckpts: dict[int, Path] = {}
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists():
        for cfg in _glob.glob(str(ckpt_root / "**" / member / "cfg.json"), recursive=True):
            d = Path(cfg).parent
            ckpts[_step_of(d)] = d

    final_dir = run_dir / member
    if (final_dir / "cfg.json").exists():
        step = max(ckpts, default=-1)
        if step < 0:
            step = _step_of(final_dir)
        return final_dir, step, True
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint for member {member!r} under {run_dir} "
            f"(looked for {member}/cfg.json and checkpoints/**/{member}/cfg.json).")
    step = max(ckpts)
    return ckpts[step], step, False


def _ensure_repo_archs_registered() -> None:
    """Register repo-local training archs (e.g. batchtopk_fp8) so raw intermediate
    checkpoints of an fp8 run can be loaded + converted. Idempotent, best-effort."""
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import architectures  # noqa: F401  (registers batchtopk_fp8 on import)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not import repo `architectures` ({e}); raw training "
              f"checkpoints (e.g. batchtopk_fp8) may fail to load.")


def load_inference_sae(ckpt_dir, device="cuda", dtype="bfloat16"):
    """Load an inference-ready SAE from ``ckpt_dir``.

    Inference archs (jumprelu/topk/standard — the final export) load directly. A raw
    ``batchtopk``/``batchtopk_fp8`` training checkpoint (intermediate) is loaded as a
    TrainingSAE and converted to its jumprelu inference form (using the learned
    threshold); the converted SAE is cached under ``<ckpt_dir>/_inference`` so reruns
    are cheap. Mirrors eval_saebench.load_inference_sae.
    """
    import gc
    from sae_lens import SAE
    from sae_lens.saes.sae import TrainingSAE

    try:
        return SAE.load_from_disk(str(ckpt_dir), device=device, dtype=dtype)
    except KeyError:
        pass  # training-only arch (batchtopk / batchtopk_fp8) -> convert below

    _ensure_repo_archs_registered()
    cache_dir = Path(ckpt_dir) / "_inference"
    if not (cache_dir / "cfg.json").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        training_sae = TrainingSAE.load_from_disk(str(ckpt_dir), device=device, dtype=dtype)
        training_sae.save_inference_model(str(cache_dir))
        del training_sae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return SAE.load_from_disk(str(cache_dir), device=device, dtype=dtype)


def load_model_and_sae(sae_dir, device="cuda", dtype="bfloat16", member=None,
                       use_latest_checkpoint=False):
    """Return ``(model, sae, meta)``. The base model is read from the SAE cfg and cached
    across SAEs that share it (so FP8 + FP16 reuse one gemma).

    ``sae_dir`` may be an exported inference-SAE dir, OR (with
    ``use_latest_checkpoint=True``) a RUN dir, in which case the latest available
    checkpoint of ``member`` is resolved (final export if present, else highest training
    step) and loaded — converting a raw training checkpoint to inference form as needed.
    The resolved load dir is recorded on ``sae._steering_load_dir`` for the caller.
    """
    _load_env()
    from transformer_lens import HookedTransformer

    if use_latest_checkpoint:
        load_dir, step, is_final = latest_checkpoint(sae_dir, member)
    else:
        load_dir, step, is_final = Path(sae_dir), _step_of(sae_dir), True

    sae = load_inference_sae(load_dir, device=device, dtype=dtype)
    sae.eval()
    sae._steering_load_dir = str(load_dir)
    sae._steering_step = step
    sae._steering_is_final = is_final
    meta = sae.cfg.metadata
    key = (meta.model_name, device, dtype)
    if key not in _MODEL_CACHE:
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        model = HookedTransformer.from_pretrained(
            meta.model_name, device=device, dtype=td,
            **(getattr(meta, "model_from_pretrained_kwargs", None) or {}),
        )
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key], sae, meta


def stream_dataset(dataset_path, seed=0, shuffle_buffer=10_000):
    """A streaming HF dataset from either an HF dataset id OR a local .jsonl(.zst)
    file/dir/glob (so a flaky network can't break feature scans — point at the cached
    Pile shard). Mirrors train_saebench_replication.build_local_dataset for local paths.
    """
    import glob as _glob
    from datasets import load_dataset

    p = Path(str(dataset_path))
    is_local = p.exists() or _glob.glob(str(dataset_path))
    if is_local:
        if p.is_dir():
            files = sorted(str(f) for pat in ("*.jsonl", "*.jsonl.zst", "*.json", "*.json.zst")
                           for f in p.rglob(pat))
        else:
            files = sorted(_glob.glob(str(dataset_path)))
        ds = load_dataset("json", data_files=files, split="train", streaming=True)
    else:
        ds = load_dataset(str(dataset_path), split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=shuffle_buffer)


def scan_features(model, sae, out_dir, dataset_path=None, n_tokens=500_000, top_n=64,
                  ctx_len=None, seqs_per_batch=8, window=8, seed=0, verbose=True):
    """Scan the dataset once and write ``concrete_features.{json,npz}`` under ``out_dir``.

    This is the all-feature pass behind steering_find_features.py, factored out so the
    notebook can (re)generate the scan for WHATEVER checkpoint it loaded — its
    per-feature max activations + top-activating tokens are checkpoint-specific.
    Returns the npz Path.
    """
    import numpy as np

    meta = sae.cfg.metadata
    dataset_path = dataset_path or meta.dataset_path
    hook_name = meta.hook_name
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    cfg_ctx = int(getattr(meta, "context_size", 1024) or 1024)
    ctx_len = ctx_len or min(cfg_ctx, 256)
    layer = layer_from_hook(hook_name)
    device = sae.W_enc.device
    enc_dtype = sae.W_enc.dtype
    tokenizer = model.tokenizer
    stop_at = (layer + 1) if layer is not None else None
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids or [])), device=device)
    bos = tokenizer.bos_token_id

    F = int(sae.cfg.d_sae)
    N = top_n
    NEG = torch.finfo(torch.float32).min
    top_vals = torch.full((F, N), NEG, device=device)
    top_toks = torch.zeros((F, N), dtype=torch.long, device=device)
    firing = torch.zeros(F, dtype=torch.long, device=device)
    glob_max = torch.full((F,), NEG, device=device)
    glob_win = torch.zeros((F, 2 * window + 1), dtype=torch.long, device=device)
    total_tokens = 0

    ds = stream_dataset(dataset_path, seed=seed)
    take = ctx_len - int(prepend_bos)
    buf: list[int] = []
    seqs: list[list[int]] = []

    def process(batch_seqs):
        nonlocal top_vals, top_toks, firing, glob_max, glob_win, total_tokens
        tokens = torch.tensor(batch_seqs, device=device)
        B, S = tokens.shape
        _, cache = model.run_with_cache(tokens, names_filter=hook_name,
                                        stop_at_layer=stop_at, return_type=None)
        fa = sae.encode(cache[hook_name].reshape(B * S, -1).to(enc_dtype)).float()
        T = fa.shape[0]
        tok_flat = tokens.reshape(T)
        if special_ids.numel():
            fa[torch.isin(tok_flat, special_ids)] = 0.0
        firing += (fa > 0).sum(0)
        total_tokens += T
        cat_vals = torch.cat([top_vals, fa.t()], dim=1)
        cat_toks = torch.cat([top_toks, tok_flat[None, :].expand(F, T)], dim=1)
        top_vals, idx = torch.topk(cat_vals, N, dim=1)
        top_toks = torch.gather(cat_toks, 1, idx)
        batch_max, batch_arg = fa.max(0)
        improved = batch_max > glob_max
        if improved.any():
            pad = torch.full((B, window), tokenizer.pad_token_id or 0,
                             dtype=torch.long, device=device)
            wins = torch.cat([pad, tokens, pad], dim=1).unfold(1, 2 * window + 1, 1)
            wins = wins.reshape(T, 2 * window + 1)
            glob_max[improved] = batch_max[improved]
            glob_win[improved] = wins[batch_arg[improved]]

    torch.set_grad_enabled(False)
    nb = 0
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= take:
            seqs.append(([bos] if prepend_bos else []) + buf[:take])
            buf = buf[take:]
            if len(seqs) == seqs_per_batch:
                process(seqs)
                seqs = []
                nb += 1
                if verbose and nb % 20 == 0:
                    print(f"    batch {nb:4d}  tokens={total_tokens:,}")
                if total_tokens >= n_tokens:
                    break
        if total_tokens >= n_tokens:
            break
    if seqs and total_tokens < n_tokens:
        process(seqs)

    # top_token_share: fraction of the top-N landing on the single most common token.
    valid = top_vals > NEG / 2
    toks_masked = torch.where(valid, top_toks, torch.full_like(top_toks, -1))
    sorted_toks, _ = toks_masked.sort(dim=1)
    valid_count = valid.sum(1).clamp(min=1).float()
    run = (sorted_toks[:, 0] != -1).float()
    best = run.clone()
    for j in range(1, N):
        cur = sorted_toks[:, j]
        eq = (cur == sorted_toks[:, j - 1]) & (cur != -1)
        run = torch.where(eq, run + 1.0, (cur != -1).float())
        best = torch.maximum(best, run)
    share = best / valid_count
    freq = firing.float() / max(total_tokens, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "concrete_features.npz"
    np.savez_compressed(
        npz,
        top_token_share=share.cpu().numpy().astype("float32"),
        firing=firing.cpu().numpy().astype("int64"),
        firing_freq=freq.cpu().numpy().astype("float32"),
        max_activation=glob_max.cpu().numpy().astype("float32"),
        top_toks=top_toks.cpu().numpy().astype("int32"),
        top_vals=top_vals.cpu().numpy().astype("float32"),
        glob_win=glob_win.cpu().numpy().astype("int32"),
        total_tokens=np.int64(total_tokens),
        window=np.int64(window),
    )
    (out_dir / "concrete_features.json").write_text(json.dumps({
        "model_name": meta.model_name, "hook_name": hook_name, "d_sae": F,
        "total_tokens": total_tokens, "top_n": N, "dataset_path": str(dataset_path),
    }, indent=2))
    if verbose:
        print(f"  scan complete ({total_tokens:,} tokens) -> {npz}")
    return npz


def ensure_feature_scan(model, sae, scan_dir, force=False, **scan_kwargs):
    """Make sure ``<scan_dir>/steering/concrete_features.npz`` exists for this SAE,
    running scan_features if it's missing (or ``force``). Returns the steering dir."""
    steer_dir = Path(scan_dir) / "steering"
    npz = steer_dir / "concrete_features.npz"
    if force or not npz.exists():
        print(f"  [scan] no feature scan at {npz} — scanning this checkpoint ...")
        scan_features(model, sae, steer_dir, **scan_kwargs)
    return steer_dir


def _scan_npz(sae_dir) -> Path:
    return Path(sae_dir) / "steering" / "concrete_features.json"


def feature_max_act(sae_dir, feature: int) -> float:
    """Max activation of ``feature`` over the scan dataset (from the .npz)."""
    import numpy as np
    npz = _scan_npz(sae_dir).with_suffix(".npz")
    d = np.load(npz)
    return float(d["max_activation"][feature])


def find_feature_by_keyword(sae_dir, keyword: str, model_name: str | None = None,
                            top_k: int = 5, min_freq: float = 2e-4,
                            max_freq: float = 0.1, whole_word: bool = False):
    """Find the feature whose top-activating token class best matches ``keyword``.

    Used to locate "the same concept" feature INDEPENDENTLY in each SAE (FP8 vs FP16),
    since the two SAEs have unrelated feature indices. Returns a list of dicts ranked by
    how strongly the keyword appears among the feature's most-common top tokens.

    ``whole_word=True`` matches the keyword only at token boundaries (regex ``\\b``), which
    avoids spurious substring hits ("war" in "toward"/"warm"/"award", "cell" in "excellent")
    that otherwise surface polysemantic, weak-to-steer features. Recommended for short or
    common keywords.
    """
    import re

    import numpy as np
    npz = _scan_npz(sae_dir).with_suffix(".npz")
    meta = json.loads(_scan_npz(sae_dir).read_text()) if _scan_npz(sae_dir).exists() else {}
    model_name = model_name or meta.get("model_name", "google/gemma-2-2b")

    _load_env()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    d = np.load(npz)
    top_toks = d["top_toks"]
    top_vals = d["top_vals"]
    freq = d["firing_freq"]
    maxact = d["max_activation"]
    F, N = top_toks.shape
    valid = top_vals > (np.finfo(np.float32).min / 2)

    kw = keyword.strip().lower()
    kw_re = re.compile(rf"\b{re.escape(kw)}\b") if whole_word else None
    hits = []
    band = (freq >= min_freq) & (freq <= max_freq)
    for f in np.nonzero(band)[0]:
        toks_f = top_toks[f][valid[f]]
        if toks_f.size == 0:
            continue
        uniq, counts = np.unique(toks_f, return_counts=True)
        order = np.argsort(-counts)
        # fraction of the top-N activations whose token text matches the keyword
        match = 0
        klass = []
        for i in order[:8]:
            s = tok.decode([int(uniq[i])])
            klass.append(s)
            txt = s.strip().lower()
            matched = bool(kw_re.search(txt)) if whole_word else (kw in txt)
            if matched:
                match += int(counts[i])
        score = match / max(int(counts.sum()), 1)
        if score > 0:
            hits.append({"feature": int(f), "match_share": round(score, 3),
                         "firing_freq": round(float(freq[f]), 6),
                         "max_activation": round(float(maxact[f]), 2),
                         "token_class": klass[:6]})
    hits.sort(key=lambda h: (h["match_share"], h["max_activation"]), reverse=True)
    return hits[:top_k]


def select_concepts(sae_dirs: dict, candidates: dict, min_match_share: float = 0.25,
                    min_max_act: float = 5.0, require_all: bool = True,
                    top_n: int | None = None, whole_word: bool = True):
    """Data-driven concept picker — the fix for hand-guessed keywords steering weakly.

    Instead of trusting an a-priori keyword list, this *validates* each candidate against the
    actual feature scans and keeps only concepts that resolve to a clean, strongly-matched,
    high-``max_act`` feature in (by default) EVERY SAE — so the FP8↔FP16 comparison is fair
    and every concept you steer is one the SAEs actually represent well.

    Args:
      sae_dirs:   ``{name -> scan dir}`` (each needs a feature scan / ``concrete_features.npz``).
      candidates: ``{concept -> [target tokens]}`` pool to consider.
      min_match_share: keep a concept only if its best feature's keyword match-share clears
                       this in the relevant SAEs (raise it to be stricter / more monosemantic).
      min_max_act: also require the matched feature's dataset-max activation to clear this
                   (low ``max_act`` features steer weakly since we clamp to ``x * max_act``).
      require_all: keep only concepts strong in ALL SAEs (True) or in ANY SAE (False).
      top_n:       cap the result to the strongest N concepts.
      whole_word:  use boundary matching (recommended) to avoid spurious substring hits.

    Returns ``(chosen, report_rows)``: ``chosen`` is ``{concept -> target tokens}`` ordered by
    strength (min over SAEs of ``match_share * max_act``); ``report_rows`` is a flat list of
    per-(concept, SAE) dicts for inspection (display as a DataFrame).
    """
    report_rows = []
    keep: dict[str, tuple[float, list]] = {}
    for c, targets in candidates.items():
        per = {}
        for name, sdir in sae_dirs.items():
            hits = find_feature_by_keyword(sdir, c, top_k=5, whole_word=whole_word)
            h = hits[0] if hits else None
            per[name] = h
            report_rows.append({
                "concept": c, "sae": name,
                "feature": (h["feature"] if h else None),
                "match_share": (h["match_share"] if h else 0.0),
                "max_act": (h["max_activation"] if h else 0.0),
                "token_class": (h["token_class"] if h else []),
            })
        good = {name: bool(h and h["match_share"] >= min_match_share
                           and h["max_activation"] >= min_max_act)
                for name, h in per.items()}
        keep_it = all(good.values()) if require_all else any(good.values())
        if keep_it:
            names = [n for n in sae_dirs if per[n]]
            strength = (min(per[n]["match_share"] for n in names)
                        * min(per[n]["max_activation"] for n in names))
            keep[c] = (strength, targets)
    ordered = sorted(keep.items(), key=lambda kv: kv[1][0], reverse=True)
    chosen = {c: v[1] for c, v in ordered}
    if top_n is not None:
        chosen = dict(list(chosen.items())[:top_n])
    return chosen, report_rows


_STOPWORD_TOKENS = {
    "the", "and", "of", "to", "a", "in", "is", "that", "it", "for", "on", "with", "as",
    "was", "are", "be", "this", "an", "at", "by", "or", "from", "but", "not", "they",
    "you", "he", "she", "we", "his", "her", "its", "their", "i", "his", "had", "have",
    "has", "were", "which", "will", "would", "can", "all", "one", "more", "out", "up",
    "what", "when", "who", "how", "there", "been", "than", "then", "them", "also", "into"}

# Markup / code / citation tokens that are frequent but NOT interesting concepts. These
# dominate purity-based discovery on web+LaTeX+code data (e.g. "usepackage", "ref", "pone").
_MARKUP_TOKENS = {
    "ref", "cite", "citep", "citet", "eqref", "label", "usepackage", "documentclass",
    "begin", "end", "textbf", "textit", "emph", "newcommand", "section", "subsection",
    "item", "href", "url", "footnote", "caption", "centering", "includegraphics",
    "def", "var", "let", "const", "int", "str", "char", "bool", "null", "none", "nil",
    "true", "false", "return", "import", "export", "function", "func", "class", "public",
    "private", "protected", "static", "void", "println", "printf", "print", "echo",
    "fig", "figure", "tab", "table", "eq", "align", "frac", "sqrt", "mathbb", "mathcal",
    "pone", "pmid", "doi", "isbn", "http", "https", "www", "com", "org", "net", "html",
    "php", "xml", "json", "css", "div", "span", "img", "src", "alt", "href", "px", "em",
    "td", "tr", "th", "li", "ul", "ol", "br", "hr", "nbsp", "amp", "quot", "lt", "gt"}


def discover_concept_features(sae_dirs: dict, model_name: str | None = None,
                              top_n: int = 12, min_max_act: float = 10.0,
                              min_purity: float = 0.4, min_freq: float = 2e-4,
                              max_freq: float = 0.02, min_zipf: float = 2.8,
                              max_zipf: float = 6.0, exclude=None):
    """Auto-discover clean, MONOSEMANTIC concept features shared by all SAEs — no keywords, no
    probe datasets. The data-driven generalisation of hand-picking features like
    ``mountain``/``cell``/``patient``: exactly the property that makes those steer well.

    For each SAE we score every feature by its top-token **purity** — the share of its
    top-activating tokens that land on a single dominant token — and keep the monosemantic
    (``purity >= min_purity``), high-``max_act`` ones in a sensible firing-frequency band
    (rare enough to be specific, not a stop-token). We then match features ACROSS SAEs by that
    dominant token (= the shared concept, auto-named from the token) and rank by the worst
    SAE's ``purity * max_act`` (so the concept is clean in BOTH FP16 and FP8).

    To keep concepts INTERESTING (not markup/code artifacts like ``usepackage``/``ref``/
    ``pone``), the dominant token must be a real English content word: its ``wordfreq`` Zipf
    frequency must fall in ``[min_zipf, max_zipf]`` (drops rare junk AND ultra-generic function
    words) and it must not be in a markup/code/citation blacklist (``_MARKUP_TOKENS`` plus any
    extra ``exclude``). If ``wordfreq`` isn't installed, only the blacklist is applied.

    Returns ``(chosen, report)``: ``chosen = {concept -> {name -> {feature, max_act, purity,
    token}}}`` ordered by strength; ``report`` is a flat list of dicts for display.
    """
    import numpy as np
    from transformers import AutoTokenizer

    try:
        from wordfreq import zipf_frequency as _zipf
    except Exception:  # noqa: BLE001
        _zipf = None
    block = set(_MARKUP_TOKENS) | {w.lower() for w in (exclude or [])}

    def _interesting(word: str) -> bool:
        if word in _STOPWORD_TOKENS or word in block:
            return False
        if _zipf is not None:
            z = _zipf(word, "en")
            if z < min_zipf or z > max_zipf:
                return False
        return True

    _load_env()
    names = list(sae_dirs)
    per_sae: dict = {}
    _tok_cache: dict = {}
    for name, sdir in sae_dirs.items():
        npz = _scan_npz(sdir).with_suffix(".npz")
        meta = json.loads(_scan_npz(sdir).read_text()) if _scan_npz(sdir).exists() else {}
        mn = model_name or meta.get("model_name", "google/gemma-2-2b")
        if mn not in _tok_cache:
            _tok_cache[mn] = AutoTokenizer.from_pretrained(mn)
        tk = _tok_cache[mn]
        d = np.load(npz)
        top_toks, top_vals = d["top_toks"], d["top_vals"]
        freq, maxact = d["firing_freq"], d["max_activation"]
        valid = top_vals > (np.finfo(np.float32).min / 2)
        band = (freq >= min_freq) & (freq <= max_freq) & (maxact >= min_max_act)
        best: dict = {}  # token -> best entry by purity*max_act
        for f in np.nonzero(band)[0]:
            ids = top_toks[f][valid[f]]
            if ids.size == 0:
                continue
            uniq, counts = np.unique(ids, return_counts=True)
            j = int(counts.argmax())
            purity = float(counts[j] / counts.sum())
            if purity < min_purity:
                continue
            s = tk.decode([int(uniq[j])]).strip()
            key = s.lower()
            if len(key) < 3 or not key.isalpha() or not _interesting(key):
                continue
            score = purity * float(maxact[f])
            if key not in best or score > best[key]["score"]:
                best[key] = {"feature": int(f), "purity": round(purity, 3),
                             "max_act": round(float(maxact[f]), 2), "token": s,
                             "score": score}
        per_sae[name] = best

    shared = (set.intersection(*[set(per_sae[n]) for n in names]) if per_sae else set())
    rows = []
    for key in shared:
        entry = {n: per_sae[n][key] for n in names}
        rows.append((min(entry[n]["score"] for n in names), key, entry))
    rows.sort(reverse=True, key=lambda r: r[0])

    chosen, report = {}, []
    for score, key, entry in rows[:top_n]:
        label = entry[names[0]]["token"].strip()
        chosen[label] = entry
        row = {"concept": label, "min_score": round(score, 1)}
        for n in names:
            row[f"{n}_feat"] = entry[n]["feature"]
            row[f"{n}_purity"] = entry[n]["purity"]
            row[f"{n}_max"] = entry[n]["max_act"]
        report.append(row)
    return chosen, report


# SAEBench class ids -> human concept names (from sae_bench.dataset_info, which states these
# are hardcoded and stable). bias_in_bios classes are PROFESSIONS; amazon_1and5 classes are
# product CATEGORIES; the _sentiment variant is 1-star/5-star reviews.
_PROFESSIONS = {
    "0": "accountant", "1": "architect", "2": "attorney", "3": "chiropractor",
    "4": "comedian", "5": "composer", "6": "dentist", "7": "dietitian", "8": "dj",
    "9": "filmmaker", "10": "interior designer", "11": "journalist", "12": "model",
    "13": "nurse", "14": "painter", "15": "paralegal", "16": "pastor",
    "17": "personal trainer", "18": "photographer", "19": "physician", "20": "poet",
    "21": "professor", "22": "psychologist", "23": "rapper", "24": "software engineer",
    "25": "surgeon", "26": "teacher", "27": "yoga teacher"}
_AMAZON_CATS = {
    "0": "beauty products", "1": "toys and games", "2": "cell phones",
    "3": "industrial and scientific", "5": "musical instruments", "6": "electronics",
    "11": "office products", "14": "sports and outdoors", "15": "home and kitchen",
    "19": "video games", "24": "books", "26": "CDs and vinyl"}
SPARSE_PROBE_LABELS: dict = {
    "Helsinki-NLP/europarl": {"en": "English", "fr": "French", "de": "German",
                              "es": "Spanish", "nl": "Dutch", "it": "Italian",
                              "da": "Danish", "fi": "Finnish", "pl": "Polish",
                              "sv": "Swedish", "pt": "Portuguese", "el": "Greek"},
    "fancyzhx/ag_news": {"0": "World news", "1": "Sports", "2": "Business",
                         "3": "Science/Tech"},
    "canrager/amazon_reviews_mcauley_1and5": _AMAZON_CATS,
    "canrager/amazon_reviews_mcauley_1and5_sentiment": {"1.0": "negative review",
                                                        "5.0": "positive review"},
    "LabHC/bias_in_bios_class_set1": _PROFESSIONS,
    "LabHC/bias_in_bios_class_set2": _PROFESSIONS,
    "LabHC/bias_in_bios_class_set3": _PROFESSIONS,
    "codeparrot/github-code": {"Python": "Python code", "C": "C code", "HTML": "HTML",
                               "Java": "Java code", "PHP": "PHP code"},
}

# Δ-logprob target tokens per concept label (falls back to the label word). Languages use
# common function words; professions/categories use defining content words.
SPARSE_PROBE_TARGETS: dict = {
    "French": [" le", " la", " les", " et", " est"],
    "German": [" der", " die", " und", " ist", " nicht"],
    "Spanish": [" el", " la", " los", " que", " y"],
    "Dutch": [" de", " het", " en", " van", " is"],
    "Italian": [" il", " la", " che", " di", " è"],
    "English": [" the", " and", " of", " to", " is"],
    "Sports": [" game", " team", " season", " player", " win"],
    "Business": [" market", " company", " shares", " profit", " economy"],
    "World news": [" government", " president", " country", " war"],
    "Science/Tech": [" software", " computer", " internet", " technology"],
    "positive review": [" great", " love", " excellent", " best", " perfect"],
    "negative review": [" bad", " worst", " terrible", " disappointed", " poor"],
    "poet": [" poem", " poetry", " verse", " poet"],
    "attorney": [" attorney", " lawyer", " court", " legal"],
    "dentist": [" dentist", " teeth", " dental", " tooth"],
    "physician": [" physician", " doctor", " patient", " medical"],
    "psychologist": [" psychologist", " psychology", " therapy", " mental"],
    "nurse": [" nurse", " nursing", " patient", " ward"],
    "professor": [" professor", " university", " research", " lecture"],
    "surgeon": [" surgeon", " surgery", " surgical", " operation"],
    "teacher": [" teacher", " classroom", " students", " school"],
    "composer": [" composer", " symphony", " orchestra", " music"],
    "photographer": [" photographer", " photo", " camera", " photograph"],
    "toys and games": [" toy", " toys", " game", " games"],
    "cell phones": [" phone", " phones", " smartphone", " battery"],
    "musical instruments": [" guitar", " instrument", " strings", " sound"],
    "electronics": [" device", " electronic", " charger", " screen"],
    "books": [" book", " books", " novel", " author"],
    "Python code": [" def", " import", " print", " self"],
    "Java code": [" public", " class", " void", " static"],
    "HTML": [" <", " div", " href", " span"],
    "C code": [" int", " char", " void", " struct"],
}

# Concept-appropriate steering prompts: a neutral lead-in where the concept could plausibly
# surface, so baseline text stays generic and the steering effect is obvious.
SPARSE_PROBE_PROMPTS: dict = {
    "French": "He leaned over to the waiter and said,",
    "German": "She greeted the visitors warmly and said,",
    "Spanish": "He waved to his neighbor across the street and said,",
    "Dutch": "The shopkeeper looked up from the counter and said,",
    "Italian": "The chef came out of the kitchen and said,",
    "English": "He cleared his throat, faced the crowd, and said,",
    "Sports": "Turning on the TV last night, I caught the highlights and",
    "Business": "Reading the morning paper, I saw a headline saying that",
    "World news": "The evening broadcast opened with a report that",
    "Science/Tech": "At the conference this week, researchers announced that",
    "positive review": "Here is my honest review of the product I just bought:",
    "negative review": "Here is my honest review of the product I just bought:",
    "poet": "At the open-mic night, she stepped up to the microphone and",
    "attorney": "In the courtroom that morning, the",
    "dentist": "At my checkup this morning, the",
    "physician": "When I described my symptoms, the",
    "psychologist": "During our first session that afternoon, the",
    "nurse": "Halfway through the night shift on the ward, the",
    "professor": "At the front of the lecture hall, the",
    "surgeon": "Just before the operation began, the",
    "teacher": "At the front of the classroom that morning, the",
    "composer": "Sitting down at the piano, the",
    "photographer": "On location at sunrise, the",
    "toys and games": "I was shopping for a birthday present and",
    "cell phones": "I finally decided to upgrade my old phone, so I",
    "musical instruments": "Walking into the shop downtown, I",
    "electronics": "I opened the box my package came in and",
    "books": "Browsing the shelves at the store, I picked up",
}


def probe_concept_label(dataset: str, cls) -> str:
    """Human label for a SAEBench (dataset, class) probe concept (falls back to ``ds:cls``)."""
    return SPARSE_PROBE_LABELS.get(dataset, {}).get(str(cls), f"{dataset.split('/')[-1]}:{cls}")


def probe_concept_targets(label: str) -> list[str]:
    """Δ-logprob target tokens for a probe concept label (falls back to the label word)."""
    if label in SPARSE_PROBE_TARGETS:
        return SPARSE_PROBE_TARGETS[label]
    w = label.split(":")[-1].split()[0]
    return [" " + w, " " + w.capitalize()]


def probe_concept_prompt(label: str, default: str = "I went for a walk and") -> str:
    """Concept-appropriate steering prompt for a probe concept label (falls back to a generic
    lead-in)."""
    return SPARSE_PROBE_PROMPTS.get(label, default)


# Topic-NEUTRAL lead-ins: they presuppose no subject, so a clamped concept (whatever it is)
# surfaces naturally — which is exactly what a clean causal steering demo wants, and "aligns"
# with any auto-discovered/contrast concept rather than a topic-specific prompt that wouldn't.
_NEUTRAL_STEER_PROMPTS = [
    "The other day I started thinking about",
    "Honestly, the thing that's been on my mind lately is",
    "Let me tell you what I really want to talk about:",
    "If you asked me what matters most right now, it's",
    "I keep coming back to one thing, and it's",
    "There's something I've been meaning to bring up, and it's",
]


def neutral_steer_prompt(concept: str) -> str:
    """A topic-neutral steering lead-in, chosen deterministically per concept so a clamped
    feature fills in the subject. Use for auto-discovered / contrast concepts that have no
    hand-written prompt, so the prompt always fits whatever concept was selected."""
    idx = sum(map(ord, concept)) % len(_NEUTRAL_STEER_PROMPTS)
    return _NEUTRAL_STEER_PROMPTS[idx]


def _latest_probe_json(run_dir: Path, member: str, step: int | None = None) -> Path | None:
    """Latest (or step-matching) sparse-probing eval_results json for ``member``."""
    import glob as _glob
    base = Path(run_dir) / "saebench_eval" / member / "eval_results" / "sparse_probing"
    files = sorted(_glob.glob(str(base / "*eval_results.json")))
    if not files:
        return None
    if step is not None:
        for f in files:
            if _step_of(Path(f)) == step:
                return Path(f)
    return Path(max(files, key=lambda f: _step_of(Path(f))))


def read_sparse_probe_scores(run_dirs: dict, member: str, step: int | None = None) -> dict:
    """Read SAEBench 1-sparse probe accuracies per (dataset, class) for several SAEs.

    ``run_dirs``: ``{name -> run dir}`` (each with ``saebench_eval/<member>/eval_results/
    sparse_probing/*eval_results.json``). Returns
    ``{(dataset, cls) -> {name: {"sae": acc, "llm": acc}}}`` using the per-class
    ``sae_top_1_test_accuracy`` / ``llm_top_1_test_accuracy`` — i.e. how well a SINGLE latent
    (vs a single raw-model direction) linearly separates that concept. The principled,
    already-computed signal for "this SAE has a clean monosemantic feature for X".
    """
    out: dict = {}
    for name, run_dir in run_dirs.items():
        jf = _latest_probe_json(Path(run_dir), member, step)
        if jf is None:
            print(f"  [warn] no sparse_probing json for {name} under {run_dir}")
            continue
        u = json.loads(jf.read_text())["eval_result_unstructured"]
        for ds_key, r in u.items():
            ds = ds_key.replace("_results", "")
            s1 = r.get("sae_top_1_test_accuracy", {})
            l1 = r.get("llm_top_1_test_accuracy", {})
            for cls, acc in s1.items():
                out.setdefault((ds, cls), {})[name] = {
                    "sae": float(acc), "llm": float(l1.get(cls, float("nan")))}
    return out


def rank_probe_concepts(scores: dict, top_n: int = 10, require_all: bool = True,
                        min_sae_acc: float = 0.0, min_sae_minus_llm: float | None = None):
    """Rank probe concepts by how cleanly ONE latent captures them across all SAEs.

    Sorts ``read_sparse_probe_scores`` output by ``min_name(sae_top_1)`` (the worst SAE, so a
    high rank means BOTH FP16 and FP8 isolate it well). Optionally require the concept present
    in every SAE (``require_all``), clear ``min_sae_acc``, and have a minimum SAE-minus-LLM
    gap (``min_sae_minus_llm``) so we prefer features the SAE isolates better than any single
    raw-model direction. Returns a list of dicts ``{dataset, cls, label, min_sae, scores}``.
    """
    names = sorted({n for v in scores.values() for n in v})
    rows = []
    for (ds, cls), per in scores.items():
        if require_all and not all(n in per for n in names):
            continue
        sae_accs = [per[n]["sae"] for n in per]
        gaps = [per[n]["sae"] - per[n]["llm"] for n in per
                if per[n]["llm"] == per[n]["llm"]]  # drop NaN
        min_sae = min(sae_accs)
        if min_sae < min_sae_acc:
            continue
        if min_sae_minus_llm is not None and (not gaps or min(gaps) < min_sae_minus_llm):
            continue
        rows.append({"dataset": ds, "cls": cls, "label": probe_concept_label(ds, cls),
                     "min_sae": round(min_sae, 3),
                     "scores": {n: round(per[n]["sae"], 3) for n in per}})
    rows.sort(key=lambda r: r["min_sae"], reverse=True)
    return rows[:top_n]


def per_token_acts(model, sae, text: str, feature: int, prepend_bos=True):
    """Return (str_tokens, activations[T]) of ``feature`` on ``text``."""
    hook_name = sae.cfg.metadata.hook_name
    layer = layer_from_hook(hook_name)
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(
        tokens, names_filter=hook_name,
        stop_at_layer=(layer + 1) if layer is not None else None,
        return_type=None,
    )
    resid = cache[hook_name]
    acts = sae.encode(resid.to(sae.W_enc.dtype))[0, :, feature].float().cpu()
    str_tokens = model.to_str_tokens(tokens[0])
    return str_tokens, acts


def top_activating_examples(model, sae, features, dataset_path=None, n_examples=5,
                            n_tokens=100_000, window=12, ctx_len=128, seqs_per_batch=16,
                            seed=0):
    """Collect each feature's top-``n_examples`` max-activating contexts from the dataset.

    This is the authentic "feature visualization" (Anthropic max-activating examples):
    for each feature we stream the dataset and keep the windows where it fires hardest,
    together with the per-token activations so they can be rendered as heatmaps.

    Returns ``{feature: [{"max_act", "str_tokens", "acts", "hot"}, ...]}`` sorted by
    activation (``hot`` is the index of the peak token within the window).
    """
    hook_name = sae.cfg.metadata.hook_name
    layer = layer_from_hook(hook_name)
    meta = sae.cfg.metadata
    dataset_path = dataset_path or meta.dataset_path
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    feats = list(features)
    fidx = torch.tensor(feats, device=sae.W_enc.device)
    bos = model.tokenizer.bos_token_id

    best: dict[int, list] = {f: [] for f in feats}  # each: (val, b_tokens, slice info)
    ds = stream_dataset(dataset_path, seed=seed)

    buf: list[int] = []
    seqs: list[list[int]] = []
    produced = 0
    take = ctx_len - int(prepend_bos)

    def flush(seqs):
        nonlocal produced
        toks = torch.tensor(seqs, device=fidx.device)
        B, Sl = toks.shape
        _, cache = model.run_with_cache(
            toks, names_filter=hook_name,
            stop_at_layer=(layer + 1) if layer is not None else None, return_type=None)
        fa = sae.encode(cache[hook_name].reshape(B * Sl, -1).to(sae.W_enc.dtype))
        fa = fa[:, fidx].reshape(B, Sl, len(feats)).float()    # [B, Sl, nfeat]
        seq_max, seq_arg = fa.max(dim=1)                       # [B, nfeat] each
        for j, f in enumerate(feats):
            for b in range(B):
                val = float(seq_max[b, j])
                if val <= 0:
                    continue
                lst = best[f]
                if len(lst) >= n_examples and val <= lst[-1][0]:
                    continue
                pos = int(seq_arg[b, j])
                lo, hi = max(0, pos - window), min(Sl, pos + window + 1)
                ids = toks[b, lo:hi].tolist()
                acts = fa[b, lo:hi, j].tolist()
                lst.append((val, ids, acts, pos - lo))
                lst.sort(key=lambda t: -t[0])
                del lst[n_examples:]
        produced += B * Sl

    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        buf.extend(model.tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= take:
            seqs.append(([bos] if prepend_bos else []) + buf[:take])
            buf = buf[take:]
            if len(seqs) == seqs_per_batch:
                flush(seqs)
                seqs = []
                if produced >= n_tokens:
                    break
        if produced >= n_tokens:
            break
    if seqs and produced < n_tokens:
        flush(seqs)

    out = {}
    for f in feats:
        out[f] = [{"max_act": v, "str_tokens": model.to_str_tokens(torch.tensor(ids)),
                   "acts": acts, "hot": hot}
                  for (v, ids, acts, hot) in best[f]]
    return out


def token_heatmap_html(str_tokens, acts, max_act=None, title="", color="255,127,14"):
    """Anthropic 'scaling monosemanticity'-style colored-token heatmap (HTML string).

    Each token's background opacity is proportional to the feature's activation there,
    normalized by ``max_act`` (defaults to the local max). ``color`` is an "r,g,b" str.
    """
    import torch as _t
    a = acts.tolist() if isinstance(acts, _t.Tensor) else list(acts)
    m = max_act if max_act else (max(a) if a else 1.0)
    m = m or 1.0
    spans = []
    for tok, val in zip(str_tokens, a):
        alpha = max(0.0, min(1.0, val / m))
        # Keep real spaces (visible, but breakable thanks to pre-wrap on the container);
        # using &nbsp; here is what made long lines overflow the cell.
        safe = html.escape(tok).replace("\n", "<br>")
        spans.append(
            f'<span title="{val:.2f}" style="background-color:rgba({color},{alpha:.3f});'
            f'border-radius:2px">{safe}</span>'
        )
    head = (f'<div style="font:13px monospace;margin:2px 0 6px;white-space:normal">'
            f'<b>{html.escape(title)}</b> '
            f'<span style="color:#888">(max={m:.2f})</span></div>') if title else ""
    return (f'<div style="font:13px/1.8 monospace;background:#fff;color:#111;padding:8px;'
            f'border:1px solid #ddd;border-radius:4px;white-space:pre-wrap;'
            f'overflow-wrap:anywhere;word-break:break-word;max-width:100%">{head}'
            + "".join(spans) + "</div>")


def feature_direction(sae, feature: int) -> torch.Tensor:
    """The residual-stream vector added per UNIT of feature ``feature``'s activation.

    Computed as ``decode(e_feature) - decode(0)`` so it is exactly the SAE's own decoder
    contribution for one unit of activation — automatically correct whether or not the SAE
    rescales activations by decoder norm, and independent of ``b_dec``. This is the right
    direction for clamping (using the raw ``W_dec`` row would be off by ``||W_dec_i||``
    whenever the SAE uses decoder-norm rescaling, as the BatchTopK/JumpReLU ones do).
    """
    d_sae = sae.cfg.d_sae
    z = torch.zeros(1, d_sae, device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    e = z.clone()
    e[0, feature] = 1.0
    return (sae.decode(e) - sae.decode(z))[0].detach()


def make_clamp_hook(sae, feature: int, target: float, direction: torch.Tensor,
                    reconstruct: bool = False):
    """Forward hook that sets ``feature``'s latent to ``target`` at every position.

    Both modes first encode the residual to get the latents ``z`` and set ``z[feature] =
    target``; they differ in what they keep around it:

    * ``reconstruct=False`` (default, **error-preserving**): returns
      ``resid + (target - z_f) * direction``. Because ``decode`` is affine this is
      *identically* ``decode(z_with_f=target) + (resid - decode(z))`` — i.e. set the latent
      to ``target`` and reconstruct, but ADD BACK the SAE reconstruction error so the rest of
      the residual (other features + unexplained part) is untouched. This is the standard
      steering choice and keeps generations coherent.
    * ``reconstruct=True`` (**pure reconstruction**): returns ``decode(z_with_f=target)`` —
      the residual is replaced entirely by the SAE's reconstruction with the feature pinned
      to ``target``. This drops the reconstruction error and is a more destructive edit;
      use it only if you explicitly want the model to see *only* what the SAE can represent.

    ``direction`` is ``decode(e_f) - decode(0)`` (correct even under decoder-norm rescaling).
    Note: re-encoding the output will not read back exactly ``target`` (encoder∘decoder is
    not identity); the clamp pins the *injected contribution*, not the re-encoded value.
    """
    def hook(resid, hook):  # noqa: ARG001
        z = sae.encode(resid.to(sae.W_enc.dtype))
        cur = z[..., feature].to(resid.dtype)  # [b,pos]
        if reconstruct:
            z = z.clone()
            z[..., feature] = target
            return sae.decode(z).to(resid.dtype)
        return resid + (target - cur).unsqueeze(-1) * direction.to(resid.dtype)
    return hook


def steer_generate(model, sae, feature: int, prompt: str, x: float, max_act: float,
                   n_samples=3, max_new_tokens=48, seed=0,
                   temperature=1.0, top_p=0.3, freq_penalty=1.0):
    """Generate continuations with ``feature`` CLAMPED to ``x * max_act`` (i.e. x times its
    largest activation seen in the dataset). x>0 amplifies the concept, x<0 suppresses it,
    x==0 applies NO hook (the genuine unsteered baseline).

    The clamp is held THROUGHOUT generation: the hook is registered for the whole
    ``model.generate`` call so it re-clamps the feature at *every* newly generated position
    (with ``use_past_kv_cache=True`` each incremental step's residual is clamped, and that
    clamp propagates into the cached K/V that future tokens attend to). Use
    :func:`clamp_fidelity` to verify the activation actually holds at the target.
    """
    hook_name = sae.cfg.metadata.hook_name
    fwd_hooks = []
    if x != 0.0:
        direction = feature_direction(sae, feature)
        fwd_hooks = [(hook_name, make_clamp_hook(sae, feature, float(x) * float(max_act),
                                                 direction))]
    torch.manual_seed(seed)
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        toks = model.to_tokens([prompt] * n_samples)
        out = model.generate(
            input=toks, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, freq_penalty=freq_penalty,
            stop_at_eos=False, verbose=False, use_past_kv_cache=True,
        )
    return [model.to_string(o[1:]) for o in out]


@torch.no_grad()
def clamp_fidelity(model, sae, feature: int, text: str, x: float, max_act: float):
    """Calibration diagnostic for the clamp across a whole sequence (FP8 vs FP16).

    The clamp injects ``target * direction`` into the residual (``target = x * max_act``),
    so the *physical* size of the intervention is ``inject_norm = |target| * ||direction||``.
    Compared to the residual's own norm this gives ``rel_strength`` — the meaningful,
    encoder-free measure of "how hard are we pushing", directly comparable between SAEs. If
    FP8's ``rel_strength`` is smaller than FP16's at the same ``x``, the *same* x is a weaker
    push there (a calibration gap), a prime suspect for FP8 steering looking worse — usually
    because FP8's ``max_act`` or decoder-row norm differs.

    We also report the *re-encoded* readback (``reencode_mean`` / ``reencode_ratio``): the
    SAE will NOT read the clamped residual back as exactly ``target`` because encoder∘decoder
    is not identity (off by ``gain = W_enc[:,f]·direction``); treat this as secondary, a hint
    about encoder geometry / feature splitting rather than clamp correctness.

    Returns a dict; ``ratio``/``achieved_*`` are kept as aliases of the re-encode fields for
    backward compatibility.
    """
    hook_name = sae.cfg.metadata.hook_name
    layer = layer_from_hook(hook_name)
    target = float(x) * float(max_act)
    direction = feature_direction(sae, feature)
    dir_norm = float(direction.float().norm())
    clamp = make_clamp_hook(sae, feature, target, direction)
    captured: dict[str, torch.Tensor] = {}

    def cap_hook(resid, hook):
        captured["resid_norm"] = resid.float().norm(dim=-1).mean().detach().cpu()
        clamped = clamp(resid, hook)
        captured["act"] = sae.encode(
            clamped.to(sae.W_enc.dtype))[..., feature].float().detach().cpu()
        return clamped

    toks = model.to_tokens(text)
    model.reset_hooks()
    with model.hooks(fwd_hooks=[(hook_name, cap_hook)]):
        model(toks, stop_at_layer=(layer + 1) if layer is not None else None,
              return_type=None)
    a = captured["act"].reshape(-1)
    resid_norm = float(captured["resid_norm"])
    inject_norm = abs(target) * dir_norm
    gain = float((sae.W_enc[:, feature].float()
                  @ direction.float().to(sae.W_enc.device)))
    reencode_mean = float(a.mean())
    reencode_ratio = (reencode_mean / (target + 1e-9)) if target else float("nan")
    return {
        "target": target,
        "dir_norm": dir_norm,
        "inject_norm": inject_norm,          # ||target * direction|| added to the residual
        "resid_norm": resid_norm,            # typical residual norm at the hook
        "rel_strength": inject_norm / (resid_norm + 1e-9),  # the cross-SAE calibration number
        "reencode_mean": reencode_mean,      # SAE's readback (NOT expected == target)
        "reencode_ratio": reencode_ratio,
        "gain": gain,                        # W_enc[:,f]·direction (encoder∘decoder gain)
        # backward-compatible aliases
        "achieved_mean": reencode_mean,
        "achieved_min": float(a.min()),
        "ratio": reencode_ratio,
    }


@torch.no_grad()
def _steer_generate_ids(model, sae, feature, prompt, x, max_act, n_samples, max_new_tokens,
                        seed, temperature, top_p, freq_penalty):
    """Like :func:`steer_generate`, but returns ``(continuation_id_lists, full_texts)`` so
    callers get the generated tokens WITHOUT the prompt (``steer_generate`` returns the
    prompt+continuation string). The clamp is held throughout generation."""
    hook_name = sae.cfg.metadata.hook_name
    fwd_hooks = []
    if x != 0.0:
        direction = feature_direction(sae, feature)
        fwd_hooks = [(hook_name, make_clamp_hook(sae, feature, float(x) * float(max_act),
                                                 direction))]
    torch.manual_seed(seed)
    model.reset_hooks()
    toks = model.to_tokens([prompt] * n_samples)
    plen = toks.shape[1]
    with model.hooks(fwd_hooks=fwd_hooks):
        out = model.generate(
            input=toks, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, freq_penalty=freq_penalty,
            stop_at_eos=False, verbose=False, use_past_kv_cache=True,
        )
    cont_ids = [o[plen:].tolist() for o in out]
    texts = [model.to_string(o[1:]) for o in out]
    return cont_ids, texts


def _target_token_ids(model, target_tokens) -> list[int]:
    """Resolve a list of target strings (e.g. ``[" mountain", " Mount"]``) to single-token
    ids, dropping any that don't map to exactly one token."""
    ids = []
    for t in target_tokens:
        try:
            ids.append(model.to_single_token(t))
        except Exception:  # noqa: BLE001  (multi-token / OOV target -> skip)
            continue
    return sorted(set(ids))


@torch.no_grad()
def _continuation_target_mass(model, sae, feature, prompt, continuation_ids, target_ids,
                              x, max_act):
    """Mean next-token prob mass on ``target_ids`` over the continuation, teacher-forced
    with the clamp held throughout (x==0 => no clamp). Measures the concept's pull at
    EVERY generated position, not just the first."""
    hook_name = sae.cfg.metadata.hook_name
    pre = model.to_tokens(prompt)[0].tolist()
    ids = torch.tensor([pre + list(continuation_ids)], device=sae.W_enc.device)
    p0 = len(pre)
    fwd_hooks = []
    if x != 0.0:
        direction = feature_direction(sae, feature)
        fwd_hooks = [(hook_name, make_clamp_hook(sae, feature, float(x) * float(max_act),
                                                 direction))]
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(ids)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    # positions p0-1 .. end-1 predict the continuation tokens
    pred = logp[p0 - 1:-1]
    if pred.shape[0] == 0:
        return float("nan")
    mass = pred[:, target_ids].logsumexp(dim=-1)  # log prob mass on target set per position
    return float(mass.mean())


@torch.no_grad()
def steering_metric(model, sae, feature: int, prompt: str, target_tokens, max_act: float,
                    x: float = 4.0, n_samples: int = 4, max_new_tokens: int = 40, seed: int = 0,
                    temperature: float = 1.0, top_p: float = 0.3, freq_penalty: float = 1.0):
    """Quantitative steering-performance metric for ONE feature (comparable FP8 vs FP16).

    Generates ``n_samples`` continuations at the steered ``x`` and at the unsteered baseline
    (x=0), the clamp held THROUGHOUT generation in the steered pass, and reports:

      * ``target_lp_delta``   Δ mean log prob-mass the model puts on the concept's target
                              tokens, measured at every generated position (steered minus
                              baseline). The headline "did steering inject the concept?".
      * ``target_rate``/``baseline_rate`` fraction of generated tokens that ARE a target
        token (a hard behavioural readout), and their delta ``target_rate_delta``.
      * ``fluency_nll``       mean per-token NLL of the steered text under the UNSTEERED
                              model (higher = less coherent; flags steering that wins the
                              concept only by emitting degenerate text).
      * ``clamp_ratio``       achieved/target activation on the steered text (from
                              :func:`clamp_fidelity`) — the calibration check.
      * ``score``             ``target_lp_delta`` minus a soft fluency penalty, so a high
                              score requires the concept to come in WITHOUT breaking text.

    All randomness is seeded so FP8 and FP16 see the same sampling draw.
    """
    target_ids = _target_token_ids(model, target_tokens)
    if not target_ids:
        raise ValueError(f"No single-token targets among {target_tokens!r}.")
    gen_kw = dict(n_samples=n_samples, max_new_tokens=max_new_tokens, seed=seed,
                  temperature=temperature, top_p=top_p, freq_penalty=freq_penalty)

    steered_ids, steered_txt = _steer_generate_ids(model, sae, feature, prompt, x, max_act,
                                                   **gen_kw)
    baseline_ids, _ = _steer_generate_ids(model, sae, feature, prompt, 0.0, max_act, **gen_kw)
    tset = set(target_ids)

    def _rate(id_lists):
        tot = sum(len(c) for c in id_lists) or 1
        hit = sum(sum(1 for t in c if t in tset) for c in id_lists)
        return hit / tot

    steered_lp = float(np.mean([
        _continuation_target_mass(model, sae, feature, prompt, c, target_ids, x, max_act)
        for c in steered_ids if c]))
    baseline_lp = float(np.mean([
        _continuation_target_mass(model, sae, feature, prompt, c, target_ids, 0.0, max_act)
        for c in baseline_ids if c]))

    # fluency: NLL of the steered continuation under the UNSTEERED model
    nlls = []
    for c in steered_ids:
        if not c:
            continue
        nlls.append(-_continuation_target_mass_self(model, prompt, c))
    fluency_nll = float(np.mean(nlls)) if nlls else float("nan")

    # steered_txt already includes the prompt + continuation; measure clamp fidelity over it.
    fid = clamp_fidelity(model, sae, feature, steered_txt[0] if steered_txt else prompt,
                         x, max_act)
    target_rate = _rate(steered_ids)
    baseline_rate = _rate(baseline_ids)
    lp_delta = steered_lp - baseline_lp
    # soft fluency penalty: only penalise text markedly less fluent than baseline.
    score = lp_delta - 0.25 * max(0.0, fluency_nll - _baseline_fluency(model, baseline_ids, prompt))
    return {
        "x": x, "feature": feature,
        "target_lp_delta": lp_delta,
        "steered_target_lp": steered_lp, "baseline_target_lp": baseline_lp,
        "target_rate": target_rate, "baseline_rate": baseline_rate,
        "target_rate_delta": target_rate - baseline_rate,
        "fluency_nll": fluency_nll,
        "inject_rel": fid["rel_strength"],     # residual-relative push size (calibration)
        "reencode_ratio": fid["reencode_ratio"],
        "clamp_gain": fid["gain"],
        "clamp_ratio": fid["ratio"],           # alias of reencode_ratio (back-compat)
        "score": score,
    }


@torch.no_grad()
def _continuation_target_mass_self(model, prompt, continuation_ids):
    """Mean per-token log-prob the UNSTEERED model assigns to its own continuation
    (used as a fluency/coherence readout; higher = more fluent)."""
    bos_pre = model.to_tokens(prompt)[0].tolist()
    ids = torch.tensor([bos_pre + list(continuation_ids)], device=model.cfg.device)
    p0 = len(bos_pre)
    model.reset_hooks()
    logits = model(ids)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    pred = logp[p0 - 1:-1]
    tgt = ids[0, p0:]
    if pred.shape[0] == 0:
        return float("nan")
    return float(pred.gather(1, tgt[:, None]).mean())


def _baseline_fluency(model, baseline_ids, prompt):
    vals = [-_continuation_target_mass_self(model, prompt, c) for c in baseline_ids if c]
    return float(np.mean(vals)) if vals else 0.0


def find_concept_features(sae_dir, keyword, top_k=4, **kw):
    """Top-``top_k`` keyword-matching features (not just the single best).

    FP8 training tends to *split* a concept across several latents; steering one of them
    then captures less of the concept than the single clean FP16 latent. Returning several
    lets the caller steer along their COMBINED decoder direction (:func:`concept_direction`)
    for a fairer, stronger intervention. Thin wrapper over
    :func:`find_feature_by_keyword`.
    """
    return find_feature_by_keyword(sae_dir, keyword, top_k=top_k, **kw)


def concept_direction(sae, features, weights=None, normalize=True):
    """Combined residual-space direction of several concept features (optionally weighted).

    ``sum_i w_i * feature_direction(feat_i)`` — the multi-feature analogue of a single
    feature's steering vector, for use with :func:`steer_generate_dir`. Defaults to an
    equal-weight sum of the given features' decoder directions.
    """
    feats = list(features)
    if not feats:
        raise ValueError("concept_direction: no features given.")
    w = weights if weights is not None else [1.0] * len(feats)
    device, dtype = sae.W_dec.device, sae.W_dec.dtype
    d = None
    for f, wi in zip(feats, w):
        contrib = float(wi) * feature_direction(sae, int(f)).to(device, dtype)
        d = contrib if d is None else d + contrib
    norm = d.norm()
    if normalize and float(norm) > 0:
        d = d / (norm + 1e-8)
    return d.detach()


def target_logprob(model, sae, feature: int, prompt: str, target_tokens: list[str],
                   x: float, max_act: float):
    """Mean log-prob of ``target_tokens`` as the next token after ``prompt`` while
    ``feature`` is clamped to x*max_act. Cheap quantitative steering metric (no OpenAI)."""
    hook_name = sae.cfg.metadata.hook_name
    tids = [model.to_single_token(t) for t in target_tokens]
    fwd_hooks = []
    if x != 0.0:
        direction = feature_direction(sae, feature)
        fwd_hooks = [(hook_name, make_clamp_hook(sae, feature, float(x) * float(max_act),
                                                 direction))]
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(model.to_tokens(prompt))[0, -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[tids].mean())


# ---------------------------------------------------------------------------
# A "deceptive" feature from TruthfulQA
#
# Instead of locating a feature by a keyword (find_feature_by_keyword), we find the
# latent whose activation best DISCRIMINATES false answers from true ones on
# TruthfulQA — i.e. a univariate linear probe for deception over the SAE basis. The
# top feature is then steered like any other concept feature (clamp to x*max_act).
# ---------------------------------------------------------------------------

ANSWER_TEMPLATE = "Q: {q}\nA:"  # answer text is appended with a leading space


def truthfulqa_statements(n_questions=None, correct_per_q=2, incorrect_per_q=2,
                          dataset_name="truthful_qa", seed=0):
    """Build ``(question, answer, label)`` triples from TruthfulQA (generation split).

    ``label`` is 1 for a deceptive (incorrect) answer and 0 for a truthful one. Up to
    ``correct_per_q`` / ``incorrect_per_q`` answers are taken per question; ``n_questions``
    caps the number of questions (None = all, shuffled by ``seed``).
    """
    import random

    from datasets import load_dataset

    # The TruthfulQA id has both a legacy script-based form and a newer parquet form; try
    # the requested id, then a couple of fallbacks / trust_remote_code, so this works
    # across `datasets` versions without the caller worrying about it.
    candidates, errors = [dataset_name, "truthfulqa/truthful_qa", "truthful_qa"], []
    ds = None
    for cand in dict.fromkeys(candidates):  # de-dup, keep order
        for kw in ({}, {"trust_remote_code": True}):
            try:
                ds = load_dataset(cand, "generation", split="validation", **kw)
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{cand} {kw}: {str(e).splitlines()[0][:80]}")
        if ds is not None:
            break
    if ds is None:
        raise RuntimeError("Could not load TruthfulQA (generation/validation). Tried:\n  "
                           + "\n  ".join(errors))
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    if n_questions:
        idxs = idxs[:n_questions]

    out = []
    for i in idxs:
        ex = ds[int(i)]
        q = (ex.get("question") or "").strip()
        if not q:
            continue
        corr = [a.strip() for a in (ex.get("correct_answers") or []) if a and a.strip()]
        best = (ex.get("best_answer") or "").strip()
        if best and best not in corr:
            corr = [best, *corr]
        inc = [a.strip() for a in (ex.get("incorrect_answers") or []) if a and a.strip()]
        for a in corr[:correct_per_q]:
            out.append((q, a, 0))
        for a in inc[:incorrect_per_q]:
            out.append((q, a, 1))
    return out


def find_deceptive_feature(model, sae, statements=None, n_questions=None,
                           correct_per_q=2, incorrect_per_q=2, agg="mean",
                           batch_size=16, min_fire_rate=0.02, top_k=15,
                           template=ANSWER_TEMPLATE, dataset_name="truthful_qa",
                           seed=0, verbose=True):
    """Rank SAE latents by how well they separate DECEPTIVE from TRUTHFUL answers.

    For each TruthfulQA statement ``"Q: <question>\\nA: <answer>"`` we run the base model,
    encode the residual at the SAE hook, and aggregate each feature's activation over the
    ANSWER tokens (``agg="mean"`` or ``"max"``). Accumulating per-class count/sum/sum-of-
    squares lets us score every feature by the standardized mean difference (Cohen's d)
    between the deceptive (label 1) and truthful (label 0) classes — a univariate linear
    probe for deception. A positive score means the latent fires MORE on false answers.

    Features firing on fewer than ``min_fire_rate`` of statements are excluded (tiny-
    variance latents otherwise produce spurious huge scores).

    Returns ``(rows, info)``: ``rows`` is the top-``top_k`` features as dicts
    (``feature``, ``deception_score``, ``mean_false``, ``mean_true``, ``fire_rate_false``,
    ``fire_rate_true``, ``max_act``) sorted by ``deception_score`` desc, where ``max_act``
    is the feature's max activation over the TruthfulQA answer tokens (use it to scale
    steering). ``info`` carries class sizes and the full per-feature ``max_act`` array.
    """
    import numpy as np

    meta = sae.cfg.metadata
    hook_name = meta.hook_name
    layer = layer_from_hook(hook_name)
    stop_at = (layer + 1) if layer is not None else None
    device = sae.W_enc.device
    enc_dtype = sae.W_enc.dtype
    tok = model.tokenizer
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    bos = tok.bos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    F = int(sae.cfg.d_sae)

    if statements is None:
        statements = truthfulqa_statements(n_questions, correct_per_q, incorrect_per_q,
                                           dataset_name, seed)
    prefix_tmpl = template.split("{a}")[0]

    built = []  # (ids, ans_start, label)
    for q, a, label in statements:
        prefix = prefix_tmpl.format(q=q)
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        ans_ids = tok(" " + a, add_special_tokens=False)["input_ids"]
        if not ans_ids:
            continue
        ids = ([bos] if prepend_bos else []) + pre_ids + ans_ids
        built.append((ids, (1 if prepend_bos else 0) + len(pre_ids), label))
    if not built:
        raise ValueError("TruthfulQA produced no usable statements.")

    sums = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    sqs = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    fires = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    counts = [0, 0]
    gmax = torch.zeros(F, device=device)

    torch.set_grad_enabled(False)
    for s in range(0, len(built), batch_size):
        chunk = built[s:s + batch_size]
        maxlen = max(len(ids) for ids, _, _ in chunk)
        batch = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        for r, (ids, _, _) in enumerate(chunk):
            batch[r, :len(ids)] = torch.tensor(ids, device=device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name,
                                        stop_at_layer=stop_at, return_type=None)
        resid = cache[hook_name]
        B, L, _ = resid.shape
        fa = sae.encode(resid.reshape(B * L, -1).to(enc_dtype)).float().reshape(B, L, F)
        for r, (ids, ans_start, label) in enumerate(chunk):
            seg = fa[r, ans_start:len(ids)]
            if seg.numel() == 0:
                continue
            v = seg.mean(0) if agg == "mean" else seg.max(0).values
            sums[label] += v.double()
            sqs[label] += v.double() ** 2
            fires[label] += (v > 0).double()
            counts[label] += 1
            gmax = torch.maximum(gmax, seg.max(0).values)
        if verbose and (s // batch_size) % 10 == 0:
            print(f"    probed {min(s + batch_size, len(built)):>5}/{len(built)} statements ...")

    n_t, n_d = counts
    if n_t == 0 or n_d == 0:
        raise ValueError(f"Need both classes; got truthful={n_t}, deceptive={n_d}.")
    mean_t, mean_d = sums[0] / n_t, sums[1] / n_d
    var_t = (sqs[0] / n_t - mean_t ** 2).clamp(min=0)
    var_d = (sqs[1] / n_d - mean_d ** 2).clamp(min=0)
    pooled = (((n_t - 1) * var_t + (n_d - 1) * var_d) / max(n_t + n_d - 2, 1)).clamp(min=0).sqrt()
    score = (mean_d - mean_t) / (pooled + 1e-6)
    fire_rate = (fires[0] + fires[1]) / (n_t + n_d)
    score = torch.where(fire_rate >= min_fire_rate, score,
                        torch.full_like(score, float("-inf")))

    score_np = score.cpu().numpy()
    mean_t_np, mean_d_np = mean_t.cpu().numpy(), mean_d.cpu().numpy()
    fr_t = (fires[0] / n_t).cpu().numpy()
    fr_d = (fires[1] / n_d).cpu().numpy()
    gmax_np = gmax.cpu().numpy()

    rows = []
    for f in np.argsort(-score_np)[:top_k]:
        f = int(f)
        if not np.isfinite(score_np[f]):
            break
        rows.append(dict(
            feature=f,
            deception_score=round(float(score_np[f]), 3),
            mean_false=round(float(mean_d_np[f]), 3),
            mean_true=round(float(mean_t_np[f]), 3),
            fire_rate_false=round(float(fr_d[f]), 3),
            fire_rate_true=round(float(fr_t[f]), 3),
            max_act=round(float(gmax_np[f]), 3),
        ))
    info = dict(n_true=n_t, n_false=n_d, agg=agg, dataset=dataset_name,
                n_statements=len(built), max_act=gmax_np)
    if verbose:
        top = rows[0] if rows else None
        print(f"  TruthfulQA probe: {n_t} truthful + {n_d} deceptive answers; "
              + (f"top feature {top['feature']} (d={top['deception_score']})"
                 if top else "no feature passed the firing-rate floor"))
    return rows, info


def find_concept_by_contrast(model, sae, positives, negatives, agg="mean",
                             batch_size=16, min_fire_rate=0.05, top_k=15,
                             skip_bos=True, verbose=True):
    """Find SAE latents for an ABSTRACT concept (e.g. ``deception``/``secrecy``) by contrast.

    Abstract concepts don't land on a single dominant token, so token-purity discovery and
    keyword matching miss them. Instead, supply ``positives`` (short texts that strongly
    exemplify the concept) and ``negatives`` (neutral/unrelated texts). We run the base model
    on each text, encode the residual at the SAE hook, aggregate every feature's activation
    over the text's tokens (``agg="mean"`` or ``"max"``), and rank latents by the standardized
    mean difference (Cohen's d) between the positive and negative classes — a univariate probe
    for the concept. Latents firing on < ``min_fire_rate`` of texts are dropped.

    Returns ``(rows, info)``: ``rows`` are the top-``top_k`` features as dicts (``feature``,
    ``score``, ``mean_pos``, ``mean_neg``, ``fire_rate_pos``, ``fire_rate_neg``, ``max_act``)
    sorted by ``score`` desc; ``max_act`` is the feature's max activation over the positive
    texts (use it to scale steering). Same shape/idea as ``find_deceptive_feature``.
    """
    import numpy as np

    meta = sae.cfg.metadata
    hook_name = meta.hook_name
    layer = layer_from_hook(hook_name)
    stop_at = (layer + 1) if layer is not None else None
    device = sae.W_enc.device
    enc_dtype = sae.W_enc.dtype
    tok = model.tokenizer
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    bos = tok.bos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    F = int(sae.cfg.d_sae)

    built = []  # (ids, start, label)  label 1 = positive (concept), 0 = negative
    for label, texts in ((1, positives), (0, negatives)):
        for t in texts:
            t = (t or "").strip()
            if not t:
                continue
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            ids = ([bos] if prepend_bos else []) + ids
            start = (1 if (prepend_bos and skip_bos) else 0)
            built.append((ids, start, label))
    if not built:
        raise ValueError("No usable contrast texts.")

    sums = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    sqs = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    fires = [torch.zeros(F, device=device, dtype=torch.float64) for _ in range(2)]
    counts = [0, 0]
    pmax = torch.zeros(F, device=device)   # max activation over POSITIVE texts (steer scale)

    torch.set_grad_enabled(False)
    for s in range(0, len(built), batch_size):
        chunk = built[s:s + batch_size]
        maxlen = max(len(ids) for ids, _, _ in chunk)
        batch = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        for r, (ids, _, _) in enumerate(chunk):
            batch[r, :len(ids)] = torch.tensor(ids, device=device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name,
                                        stop_at_layer=stop_at, return_type=None)
        resid = cache[hook_name]
        B, L, _ = resid.shape
        fa = sae.encode(resid.reshape(B * L, -1).to(enc_dtype)).float().reshape(B, L, F)
        for r, (ids, start, label) in enumerate(chunk):
            seg = fa[r, start:len(ids)]
            if seg.numel() == 0:
                continue
            v = seg.mean(0) if agg == "mean" else seg.max(0).values
            sums[label] += v.double()
            sqs[label] += v.double() ** 2
            fires[label] += (v > 0).double()
            counts[label] += 1
            if label == 1:
                pmax = torch.maximum(pmax, seg.max(0).values)
        if verbose and (s // batch_size) % 10 == 0:
            print(f"    probed {min(s + batch_size, len(built)):>5}/{len(built)} texts ...")

    n_neg, n_pos = counts
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes; got positive={n_pos}, negative={n_neg}.")
    mean_n, mean_p = sums[0] / n_neg, sums[1] / n_pos
    var_n = (sqs[0] / n_neg - mean_n ** 2).clamp(min=0)
    var_p = (sqs[1] / n_pos - mean_p ** 2).clamp(min=0)
    pooled = (((n_neg - 1) * var_n + (n_pos - 1) * var_p)
              / max(n_neg + n_pos - 2, 1)).clamp(min=0).sqrt()
    score = (mean_p - mean_n) / (pooled + 1e-6)
    fire_rate = (fires[0] + fires[1]) / (n_neg + n_pos)
    score = torch.where(fire_rate >= min_fire_rate, score,
                        torch.full_like(score, float("-inf")))

    score_np = score.cpu().numpy()
    mean_n_np, mean_p_np = mean_n.cpu().numpy(), mean_p.cpu().numpy()
    fr_n = (fires[0] / n_neg).cpu().numpy()
    fr_p = (fires[1] / n_pos).cpu().numpy()
    pmax_np = pmax.cpu().numpy()

    rows = []
    for f in np.argsort(-score_np)[:top_k]:
        f = int(f)
        if not np.isfinite(score_np[f]):
            break
        rows.append(dict(
            feature=f, score=round(float(score_np[f]), 3),
            mean_pos=round(float(mean_p_np[f]), 3), mean_neg=round(float(mean_n_np[f]), 3),
            fire_rate_pos=round(float(fr_p[f]), 3), fire_rate_neg=round(float(fr_n[f]), 3),
            max_act=round(float(pmax_np[f]), 3)))
    info = dict(n_pos=n_pos, n_neg=n_neg, agg=agg, n_texts=len(built), max_act=pmax_np)
    if verbose:
        top = rows[0] if rows else None
        print(f"  contrast probe: {n_pos} positive + {n_neg} negative texts; "
              + (f"top feature {top['feature']} (d={top['score']}, max_act={top['max_act']})"
                 if top else "no feature passed the firing-rate floor"))
    return rows, info


def answer_logprob(model, sae, feature, prefix, answer, x, max_act):
    """Mean per-token log-prob the model assigns to ``answer`` right after ``prefix``,
    while ``feature`` is clamped to ``x * max_act`` (``x==0`` => no steering).

    A length-normalized score for "how much the model wants to say this answer", used to
    test whether amplifying a deceptive feature makes the model prefer false answers.
    """
    hook_name = sae.cfg.metadata.hook_name
    tok = model.tokenizer
    bos = tok.bos_token_id
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ans_ids = tok(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    if not ans_ids:
        return float("nan")
    ids = torch.tensor([[bos, *pre_ids, *ans_ids]], device=sae.W_enc.device)
    ans_start = 1 + len(pre_ids)

    fwd_hooks = []
    if x != 0.0:
        direction = feature_direction(sae, feature)
        fwd_hooks = [(hook_name, make_clamp_hook(sae, feature, float(x) * float(max_act),
                                                 direction))]
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(ids)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[0, ans_start:]
    pred = logp[ans_start - 1:-1]
    return float(pred.gather(1, targets[:, None]).mean())


# ---------------------------------------------------------------------------
# Multi-feature deception probe + direction steering
#
# Deception is rarely a single latent. We fit a (sparse) logistic-regression probe over
# the SAE features (mean-pooled over answer tokens) to predict false-vs-true on a
# question-grouped train/test split, then steer along the probe's residual-space direction
# (a weighted sum of the selected features' decoder directions). The combined direction is
# both more predictive and a much stronger steering vector than any single latent.
# ---------------------------------------------------------------------------

def truthfulqa_feature_matrix(model, sae, statements, agg="mean", batch_size=16,
                              min_statements_fired=3, verbose=True):
    """Build the per-statement SAE-feature matrix for TruthfulQA.

    Runs each ``"Q: ...\\nA: ..."`` statement, encodes the SAE hook, aggregates each
    feature over the ANSWER tokens (``agg`` "mean"|"max"), and returns
    ``(X, y, groups, feats, info)``:
      * ``X``      — ``[n_statements, n_kept]`` float32 (CPU), restricted to features that
        fire in >= ``min_statements_fired`` statements (keeps the probe tractable/stable).
      * ``y``      — int labels (1 = deceptive/false, 0 = truthful).
      * ``groups`` — question id per statement (use for a leakage-free grouped split: a
        question's true & false variants must not straddle train/test).
      * ``feats``  — global SAE feature index for each column of ``X``.
      * ``info``   — ``n``, ``F``, ``kept``, per-feature ``gmax`` over answer tokens, ``agg``.
    """
    import numpy as np

    meta = sae.cfg.metadata
    hook_name = meta.hook_name
    layer = layer_from_hook(hook_name)
    stop_at = (layer + 1) if layer is not None else None
    device = sae.W_enc.device
    enc_dtype = sae.W_enc.dtype
    tok = model.tokenizer
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    bos = tok.bos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    F = int(sae.cfg.d_sae)
    prefix_tmpl = ANSWER_TEMPLATE.split("{a}")[0]

    qids: dict[str, int] = {}
    built = []  # (ids, ans_start, label, gid)
    for q, a, label in statements:
        gid = qids.setdefault(q, len(qids))
        pre_ids = tok(prefix_tmpl.format(q=q), add_special_tokens=False)["input_ids"]
        ans_ids = tok(" " + a, add_special_tokens=False)["input_ids"]
        if not ans_ids:
            continue
        ids = ([bos] if prepend_bos else []) + pre_ids + ans_ids
        built.append((ids, (1 if prepend_bos else 0) + len(pre_ids), label, gid))
    n = len(built)
    if n == 0:
        raise ValueError("No usable TruthfulQA statements for the feature matrix.")

    X = torch.zeros(n, F, dtype=torch.float32)  # CPU; restricted to fired cols below
    y = torch.zeros(n, dtype=torch.long)
    groups = torch.zeros(n, dtype=torch.long)
    gmax = torch.zeros(F, device=device)

    torch.set_grad_enabled(False)
    row = 0
    for s in range(0, n, batch_size):
        chunk = built[s:s + batch_size]
        maxlen = max(len(ids) for ids, _, _, _ in chunk)
        batch = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        for r, (ids, _, _, _) in enumerate(chunk):
            batch[r, :len(ids)] = torch.tensor(ids, device=device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name,
                                        stop_at_layer=stop_at, return_type=None)
        resid = cache[hook_name]
        B, L, _ = resid.shape
        fa = sae.encode(resid.reshape(B * L, -1).to(enc_dtype)).float().reshape(B, L, F)
        for r, (ids, ans_start, label, gid) in enumerate(chunk):
            seg = fa[r, ans_start:len(ids)]
            if seg.numel() == 0:
                continue
            v = seg.mean(0) if agg == "mean" else seg.max(0).values
            X[row] = v.cpu()
            y[row] = label
            groups[row] = gid
            gmax = torch.maximum(gmax, seg.max(0).values)
            row += 1
        if verbose and (s // batch_size) % 10 == 0:
            print(f"    feature-matrix {min(s + batch_size, n):>5}/{n} statements ...")

    Xnp = X[:row].numpy()
    y = y[:row].numpy()
    groups = groups[:row].numpy()
    fired = (Xnp > 0).sum(0)
    keep = np.nonzero(fired >= min_statements_fired)[0].astype(int)
    info = dict(n=int(row), F=F, kept=int(len(keep)), gmax=gmax.cpu().numpy(), agg=agg)
    if verbose:
        print(f"  feature matrix {row}x{F}; {len(keep)} latents fire in "
              f">= {min_statements_fired} statements (probe input width).")
    return Xnp[:, keep], y, groups, keep, info


def fit_deception_probe(X, y, groups=None, feats=None, penalty="l1", C=0.5,
                        test_frac=0.3, seed=0, standardize=True, max_select=64,
                        verbose=True):
    """Fit a logistic-regression deception probe over SAE features.

    ``X`` is ``[n, n_feat]`` (from :func:`truthfulqa_feature_matrix`), ``y`` the 1=deceptive
    labels. With ``groups`` we use a question-grouped train/test split so the held-out AUC
    reflects generalization to NEW questions (not memorized topics). ``penalty="l1"`` gives a
    sparse SET of latents; raise ``C`` for a stronger (less regularized) probe.

    Returns a dict: ``selected`` (top ``max_select`` features as
    ``{feature, weight, weight_raw}``, ``weight_raw`` mapped back to raw-activation units for
    steering), ``info`` (test/train AUC + accuracy, #nonzero), and the train/test indices.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    n = len(y)
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_frac,
                                        random_state=seed).split(X, y, groups))
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        cut = int(n * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]

    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd if standardize else X
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=5000,
                             class_weight="balanced")
    clf.fit(Xs[tr], y[tr])

    def _auc(ix):
        return (roc_auc_score(y[ix], clf.predict_proba(Xs[ix])[:, 1])
                if len(set(y[ix])) > 1 else float("nan"))

    proba_te = clf.predict_proba(Xs[te])[:, 1]
    info = dict(
        test_auc=round(float(_auc(te)), 4),
        test_acc=round(float(accuracy_score(y[te], (proba_te > 0.5).astype(int))), 4),
        train_auc=round(float(_auc(tr)), 4),
        n_train=int(len(tr)), n_test=int(len(te)),
        penalty=penalty, C=C,
    )
    w = clf.coef_[0]
    w_raw = w / sd if standardize else w  # raw-activation-space weights (for steering)
    info["n_nonzero"] = int(np.count_nonzero(w))
    order = [i for i in np.argsort(-np.abs(w)) if w[i] != 0][:max_select]
    selected = [dict(feature=int(feats[i]) if feats is not None else int(i),
                     weight=float(w[i]), weight_raw=float(w_raw[i])) for i in order]
    if verbose:
        print(f"  probe: test AUC={info['test_auc']}  test acc={info['test_acc']}  "
              f"(train AUC={info['train_auc']}); {info['n_nonzero']} nonzero latents, "
              f"steering with top {len(selected)}.")
    return dict(selected=selected, weights_raw=w_raw, feats=feats, info=info,
                train_idx=tr, test_idx=te)


def probe_direction(sae, selected, normalize=True):
    """Residual-space deception direction from a fitted probe.

    Combines the selected features' decoder directions weighted by their raw-activation
    probe weights: ``sum_i weight_raw_i * feature_direction(feat_i)``. Positive coefficients
    point toward DECEPTIVE. Returns a (optionally unit-normalized) residual vector.
    """
    if not selected:
        raise ValueError("probe_direction: empty `selected` feature list.")
    device, dtype = sae.W_dec.device, sae.W_dec.dtype
    d = None
    for s in selected:
        w = float(s.get("weight_raw", s.get("weight", 0.0)))
        if w == 0.0:
            continue
        contrib = w * feature_direction(sae, int(s["feature"])).to(device, dtype)
        d = contrib if d is None else d + contrib
    if d is None:
        raise ValueError("probe_direction: all selected weights are zero.")
    norm = d.norm()
    if normalize and float(norm) > 0:
        d = d / (norm + 1e-8)
    return d.detach()


def resid_norm_at_hook(model, sae, prompt="The capital of France is"):
    """Mean residual-stream L2 norm at the SAE hook (to calibrate steering magnitude)."""
    hook_name = sae.cfg.metadata.hook_name
    layer = layer_from_hook(hook_name)
    _, cache = model.run_with_cache(
        model.to_tokens(prompt), names_filter=hook_name,
        stop_at_layer=(layer + 1) if layer is not None else None, return_type=None)
    return float(cache[hook_name][0].norm(dim=-1).mean())


def _add_dir_hook(direction, coeff):
    def hook(resid, hook):  # noqa: ARG001
        return resid + float(coeff) * direction.to(resid.dtype)
    return hook


def steer_generate_dir(model, sae, direction, prompt, coeff, n_samples=3, max_new_tokens=48,
                       seed=0, temperature=1.0, top_p=0.3, freq_penalty=1.0):
    """Generate with ``coeff * direction`` ADDED to the residual at the SAE hook
    (``coeff==0`` => unsteered baseline). ``direction`` is typically unit-norm, so ``coeff``
    is in residual-activation units — calibrate against :func:`resid_norm_at_hook`."""
    hook_name = sae.cfg.metadata.hook_name
    fwd_hooks = [] if coeff == 0.0 else [(hook_name, _add_dir_hook(direction, coeff))]
    torch.manual_seed(seed)
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        toks = model.to_tokens([prompt] * n_samples)
        out = model.generate(
            input=toks, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, freq_penalty=freq_penalty,
            stop_at_eos=False, verbose=False, use_past_kv_cache=True,
        )
    return [model.to_string(o[1:]) for o in out]


def answer_logprob_dir(model, sae, direction, prefix, answer, coeff):
    """Mean per-token log-prob of ``answer`` after ``prefix`` with ``coeff * direction`` added
    at the SAE hook. Direction-steering analogue of :func:`answer_logprob`."""
    hook_name = sae.cfg.metadata.hook_name
    tok = model.tokenizer
    bos = tok.bos_token_id
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ans_ids = tok(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    if not ans_ids:
        return float("nan")
    ids = torch.tensor([[bos, *pre_ids, *ans_ids]], device=sae.W_enc.device)
    ans_start = 1 + len(pre_ids)
    fwd_hooks = [] if coeff == 0.0 else [(hook_name, _add_dir_hook(direction, coeff))]
    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(ids)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[0, ans_start:]
    pred = logp[ans_start - 1:-1]
    return float(pred.gather(1, targets[:, None]).mean())
