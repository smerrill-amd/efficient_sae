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


def load_model_and_sae(sae_dir, device="cuda", dtype="bfloat16"):
    """Return (model, sae, meta). The base model is read from the SAE cfg and cached
    across SAEs that share it (so FP8 + FP16 reuse one gemma)."""
    _load_env()
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    sae = SAE.load_from_disk(str(sae_dir), device=device, dtype=dtype)
    sae.eval()
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
                            max_freq: float = 0.1):
    """Find the feature whose top-activating token class best matches ``keyword``.

    Used to locate "the same concept" feature INDEPENDENTLY in each SAE (FP8 vs FP16),
    since the two SAEs have unrelated feature indices. Returns a list of dicts ranked by
    how strongly the keyword appears among the feature's most-common top tokens.
    """
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
    hits = []
    band = (freq >= min_freq) & (freq <= max_freq)
    for f in np.nonzero(band)[0]:
        toks_f = top_toks[f][valid[f]]
        if toks_f.size == 0:
            continue
        uniq, counts = np.unique(toks_f, return_counts=True)
        order = np.argsort(-counts)
        # fraction of the top-N activations whose token text contains the keyword
        match = 0
        klass = []
        for i in order[:8]:
            s = tok.decode([int(uniq[i])])
            klass.append(s)
            if kw in s.strip().lower():
                match += int(counts[i])
        score = match / max(int(counts.sum()), 1)
        if score > 0:
            hits.append({"feature": int(f), "match_share": round(score, 3),
                         "firing_freq": round(float(freq[f]), 6),
                         "max_activation": round(float(maxact[f]), 2),
                         "token_class": klass[:6]})
    hits.sort(key=lambda h: (h["match_share"], h["max_activation"]), reverse=True)
    return hits[:top_k]


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
    from datasets import load_dataset

    hook_name = sae.cfg.metadata.hook_name
    layer = layer_from_hook(hook_name)
    meta = sae.cfg.metadata
    dataset_path = dataset_path or meta.dataset_path
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    feats = list(features)
    fidx = torch.tensor(feats, device=sae.W_enc.device)
    bos = model.tokenizer.bos_token_id

    best: dict[int, list] = {f: [] for f in feats}  # each: (val, b_tokens, slice info)
    ds = load_dataset(dataset_path, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=10_000)

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


def make_clamp_hook(sae, feature: int, target: float, direction: torch.Tensor):
    """Forward hook that CLAMPS ``feature``'s activation to ``target`` at every position.

    Reads the feature's current activation via the encoder and adds
    ``(target - current) * direction`` so the feature's residual contribution becomes
    exactly ``target * direction`` — a true clamp, not an additive bump.
    """
    def hook(resid, hook):  # noqa: ARG001
        cur = sae.encode(resid.to(sae.W_enc.dtype))[..., feature].to(resid.dtype)  # [b,pos]
        return resid + (target - cur).unsqueeze(-1) * direction.to(resid.dtype)
    return hook


def steer_generate(model, sae, feature: int, prompt: str, x: float, max_act: float,
                   n_samples=3, max_new_tokens=48, seed=0,
                   temperature=1.0, top_p=0.3, freq_penalty=1.0):
    """Generate continuations with ``feature`` CLAMPED to ``x * max_act`` (i.e. x times its
    largest activation seen in the dataset). x>0 amplifies the concept, x<0 suppresses it,
    x==0 applies NO hook (the genuine unsteered baseline)."""
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
            stop_at_eos=False, verbose=False,
        )
    return [model.to_string(o[1:]) for o in out]


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
