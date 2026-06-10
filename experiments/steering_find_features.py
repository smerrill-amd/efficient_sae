#!/usr/bin/env python3
"""steering_find_features.py — Stage 0+1 of the steering experiment.

Find a SMALL set of *concrete* (token-level, monosemantic-looking) SAE features
WITHOUT spending any auto-interp / OpenAI tokens, so we only pay the LLM bill on a
handful of obviously-interpretable candidates later.

Pipeline (all free):
  Stage 0  collect feature activations of an exported inference SAE over a chunk of
           the SAE's own training distribution (the Pile by default). The base model
           is read from the SAE's cfg.json (metadata.model_name) so the model is a
           PARAMETER of the SAE, never hardcoded.
  Stage 1  score each feature by how *peaked* its top-activating-token distribution
           is. A concrete feature fires on one token / one tight token-class, so:
             top_token_share = (# of its top-N activations on its single most
                                common token) / N          (high  -> concrete)
             token_entropy   = entropy over those top-N tokens (low -> concrete)
           Dead (too rare) and ultra-dense (fires everywhere) features are gated out.

Output: a ranked JSON of candidate features with their dominant token + a sample
max-activating context, ready to eyeball before picking ~5-8 for auto-interp +
steering. A compact .npz of the per-feature stats is written alongside.

Example:
  python steering_find_features.py \
      --sae-dir results/saebench_gemma_fp8/w65536_k80 \
      --n-tokens 500000 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--sae-dir", type=Path, required=True,
                   help="Exported inference SAE dir (holds cfg.json). The base model "
                        "+ hook are read from its cfg.json.")
    p.add_argument("--model", default=None,
                   help="Override base model (default: read SAE cfg metadata.model_name).")
    p.add_argument("--dataset", default=None,
                   help="HF dataset (default: read SAE cfg metadata.dataset_path, "
                        "i.e. the distribution the SAE was trained on).")
    p.add_argument("--n-tokens", type=int, default=500_000,
                   help="Approx number of tokens to scan for activations.")
    p.add_argument("--ctx-len", type=int, default=None,
                   help="Sequence length (default: SAE cfg context_size, capped to 256 "
                        "for speed unless overridden).")
    p.add_argument("--seqs-per-batch", type=int, default=8,
                   help="Sequences per forward pass.")
    p.add_argument("--top-n", type=int, default=64,
                   help="Top-N max activations tracked per feature (the concreteness "
                        "distribution is computed over these).")
    p.add_argument("--n-candidates", type=int, default=30,
                   help="How many ranked candidates to print / keep in the JSON.")
    p.add_argument("--min-freq", type=float, default=1e-4,
                   help="Drop features firing on < this fraction of tokens (near-dead).")
    p.add_argument("--max-freq", type=float, default=0.2,
                   help="Drop features firing on > this fraction of tokens (too dense "
                        "to be a specific concept).")
    p.add_argument("--window", type=int, default=8,
                   help="Half-width (tokens) of the saved max-activating context.")
    p.add_argument("--content-only", dest="content_only", action="store_true",
                   default=True,
                   help="Only rank features whose dominant token is a content word "
                        "(alphabetic, not a stopword/special token). On by default — "
                        "concept features steer more convincingly than punctuation.")
    p.add_argument("--allow-structural", dest="content_only", action="store_false",
                   help="Include punctuation/stopword/structural features in the ranking.")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model + SAE compute dtype.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON (default: <sae-dir>/steering/concrete_features.json).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_env(project_root: Path) -> None:
    """Load .env (gemma-2-2b is gated and needs HF_TOKEN), mirroring eval_saebench.py."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def iter_token_batches(dataset_path, tokenizer, ctx_len, seqs_per_batch, n_tokens,
                       prepend_bos, seed):
    """Yield [B, ctx_len] int64 token batches streamed + packed from the dataset."""
    import torch
    from datasets import load_dataset

    ds = load_dataset(dataset_path, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    bos = tokenizer.bos_token_id
    buf: list[int] = []
    seqs: list[list[int]] = []
    produced = 0
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= ctx_len - int(prepend_bos):
            take = ctx_len - int(prepend_bos)
            seq = ([bos] if prepend_bos else []) + buf[:take]
            buf = buf[take:]
            seqs.append(seq)
            if len(seqs) == seqs_per_batch:
                batch = torch.tensor(seqs, dtype=torch.long)
                produced += batch.numel()
                yield batch
                seqs = []
                if produced >= n_tokens:
                    return


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_env(project_root)

    import numpy as np
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # --- Load the SAE; everything about the model/hook comes from its own config. ---
    sae = SAE.load_from_disk(str(args.sae_dir), device=device, dtype=args.dtype)
    sae.eval()
    meta = sae.cfg.metadata
    model_name = args.model or meta.model_name
    dataset_path = args.dataset or meta.dataset_path
    hook_name = meta.hook_name
    prepend_bos = bool(getattr(meta, "prepend_bos", True))
    cfg_ctx = int(getattr(meta, "context_size", 1024) or 1024)
    ctx_len = args.ctx_len or min(cfg_ctx, 256)
    layer = _layer_from_hook(hook_name)
    d_sae = int(sae.cfg.d_sae)

    print("=" * 72)
    print("  steering: free concrete-feature finder (Stage 0+1)")
    print(f"  sae        {args.sae_dir}")
    print(f"  model      {model_name}   (read from SAE cfg)")
    print(f"  hook       {hook_name}  (layer {layer}, d_sae={d_sae})")
    print(f"  dataset    {dataset_path}")
    print(f"  scan       ~{args.n_tokens:,} tokens @ ctx_len {ctx_len}, "
          f"top_n {args.top_n}")
    print(f"  device     {device}   dtype={args.dtype}")
    print("=" * 72)

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        **(getattr(meta, "model_from_pretrained_kwargs", None) or {}),
    )
    model.eval()
    tokenizer = model.tokenizer
    stop_at = (layer + 1) if layer is not None else None
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids or [])),
                               device=device)

    # --- Per-feature running state (all on GPU) ---------------------------------
    F = d_sae
    N = args.top_n
    NEG = torch.finfo(torch.float32).min
    top_vals = torch.full((F, N), NEG, device=device)            # top-N activations
    top_toks = torch.zeros((F, N), dtype=torch.long, device=device)
    firing = torch.zeros(F, dtype=torch.long, device=device)     # token fire count
    glob_max = torch.full((F,), NEG, device=device)              # best-ever activation
    glob_win = torch.zeros((F, 2 * args.window + 1), dtype=torch.long, device=device)
    total_tokens = 0

    torch.set_grad_enabled(False)
    n_batches_est = max(1, args.n_tokens // (args.seqs_per_batch * ctx_len))
    print(f"  ~{n_batches_est} forward batches\n")

    for bi, tokens in enumerate(iter_token_batches(
            dataset_path, tokenizer, ctx_len, args.seqs_per_batch,
            args.n_tokens, prepend_bos, args.seed)):
        tokens = tokens.to(device)
        B, S = tokens.shape
        _, cache = model.run_with_cache(
            tokens, names_filter=hook_name, stop_at_layer=stop_at,
            return_type=None,
        )
        resid = cache[hook_name].reshape(B * S, -1).to(dtype)   # [T, d_in]
        fa = sae.encode(resid).float()                          # [T, F]
        T = fa.shape[0]
        tok_flat = tokens.reshape(T)

        # Zero out activations at special-token positions (bos/eos/pad). Features that
        # only fire on bos are positional artifacts, not steerable concepts.
        if special_ids.numel():
            special_pos = torch.isin(tok_flat, special_ids)     # [T]
            fa[special_pos] = 0.0

        firing += (fa > 0).sum(0)
        total_tokens += T

        # Update per-feature top-N over (existing N) + (this batch's T) candidates.
        cat_vals = torch.cat([top_vals, fa.t()], dim=1)                  # [F, N+T]
        cat_toks = torch.cat([top_toks, tok_flat[None, :].expand(F, T)], dim=1)
        top_vals, idx = torch.topk(cat_vals, N, dim=1)
        top_toks = torch.gather(cat_toks, 1, idx)

        # Update the single best-activating context window per feature.
        batch_max, batch_arg = fa.max(0)                                 # [F]
        improved = batch_max > glob_max
        if improved.any():
            w = args.window
            pad = torch.full((B, w), tokenizer.pad_token_id or 0,
                             dtype=torch.long, device=device)
            padded = torch.cat([pad, tokens, pad], dim=1)                # [B, S+2w]
            wins = padded.unfold(1, 2 * w + 1, 1)                        # [B, S, 2w+1]
            wins = wins.reshape(T, 2 * w + 1)                            # [T, 2w+1]
            glob_max[improved] = batch_max[improved]
            glob_win[improved] = wins[batch_arg[improved]]

        if (bi + 1) % 20 == 0:
            print(f"    batch {bi + 1:4d}  tokens={total_tokens:,}")

    # --- Stage 1: concreteness scoring (free) -----------------------------------
    # top_token_share: fraction of the top-N landing on the single most common token.
    # Computed vectorized across all F features: sort each row, then the mode count
    # is the longest run of equal (valid) tokens. Invalid slots are masked to -1.
    valid = top_vals > NEG / 2                                   # [F, N]
    toks_masked = torch.where(valid, top_toks, torch.full_like(top_toks, -1))
    sorted_toks, _ = toks_masked.sort(dim=1)                     # [F, N]
    valid_count = valid.sum(1).clamp(min=1).float()             # [F]
    run = (sorted_toks[:, 0] != -1).float()                     # current run length
    best = run.clone()                                          # mode count so far
    for j in range(1, N):
        cur = sorted_toks[:, j]
        eq = (cur == sorted_toks[:, j - 1]) & (cur != -1)
        run = torch.where(eq, run + 1.0, (cur != -1).float())
        best = torch.maximum(best, run)
    share = best / valid_count

    freq = firing.float() / max(total_tokens, 1)
    # Require the feature to have fired at least top_n times so its top-N distribution
    # is a real selection (not trivially share==1 because it only fired a few times).
    enough = firing >= N
    keep = ((freq >= args.min_freq) & (freq <= args.max_freq)
            & (glob_max > NEG / 2) & enough)
    # Rank: most peaked top-token distribution first, strongest activation as tie-break.
    score = share.clone()
    score[~keep] = -1.0
    order = torch.argsort(score + 1e-6 * glob_max.clamp(min=0), descending=True)

    # Walk the ranked list, computing the dominant token lazily and (optionally)
    # keeping only content-word features, until we have n_candidates.
    STOPWORDS = {
        "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was", "were",
        "be", "been", "being", "for", "on", "at", "by", "with", "as", "that", "this",
        "it", "its", "from", "but", "not", "no", "so", "if", "then", "than", "i",
        "you", "he", "she", "they", "we", "his", "her", "their", "our", "my",
    }

    def is_content(tok_str: str) -> bool:
        s = tok_str.strip()
        return bool(s) and any(c.isalpha() for c in s) and s.lower() not in STOPWORDS

    candidates = []
    for f in order.tolist():
        if score[f] <= 0:
            break  # past the filtered region
        toks_f = top_toks[f][valid[f]]
        uniq, counts = torch.unique(toks_f, return_counts=True)
        dom = int(uniq[counts.argmax()].item())
        max_tok_text = tokenizer.decode([dom])
        if args.content_only and not is_content(max_tok_text):
            continue
        p = counts.float() / counts.sum()
        entropy = float(-(p * (p + 1e-12).log()).sum())
        win_ids = glob_win[f].tolist()
        win_text = tokenizer.decode(win_ids)
        candidates.append({
            "feature": int(f),
            "score_top_token_share": round(float(share[f]), 4),
            "token_entropy": round(entropy, 4),
            "firing_freq": round(float(freq[f]), 6),
            "max_activation": round(float(glob_max[f]), 4),
            "dominant_token": max_tok_text,
            "dominant_token_id": int(dom),
            "max_context": win_text,
        })
        if len(candidates) >= args.n_candidates:
            break

    out = args.output or (args.sae_dir / "steering" / "concrete_features.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sae_dir": str(args.sae_dir),
        "model_name": model_name,
        "hook_name": hook_name,
        "d_sae": d_sae,
        "total_tokens": total_tokens,
        "top_n": N,
        "filters": {"min_freq": args.min_freq, "max_freq": args.max_freq},
        "candidates": candidates,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Save the raw per-feature material so the ranking heuristic can be re-tuned
    # OFFLINE (different freq bands, token-class vs single-token, etc.) without
    # re-running the model — the expensive part. See steering_rank_features.py.
    npz = out.with_suffix(".npz")
    np.savez_compressed(
        npz,
        top_token_share=share.cpu().numpy().astype("float32"),
        firing=firing.cpu().numpy().astype("int64"),
        firing_freq=freq.cpu().numpy().astype("float32"),
        max_activation=glob_max.cpu().numpy().astype("float32"),
        top_toks=top_toks.cpu().numpy().astype("int32"),     # [F, N] top-activating tokens
        top_vals=top_vals.cpu().numpy().astype("float32"),   # [F, N] their activations
        glob_win=glob_win.cpu().numpy().astype("int32"),     # [F, 2w+1] max context window
        total_tokens=np.int64(total_tokens),
        window=np.int64(args.window),
    )

    print("\n  Top concrete-feature candidates "
          "(eyeball these; pick ~5-8 for auto-interp + steering):\n")
    print(f"  {'feat':>6}  {'share':>5}  {'ent':>5}  {'freq':>8}  "
          f"{'maxact':>7}  token / context")
    for c in candidates:
        ctx = c["max_context"].replace("\n", " ")[:70]
        print(f"  {c['feature']:>6}  {c['score_top_token_share']:>5.2f}  "
              f"{c['token_entropy']:>5.2f}  {c['firing_freq']:>8.5f}  "
              f"{c['max_activation']:>7.2f}  "
              f"[{c['dominant_token']!r}]  {ctx}")
    print(f"\n  Wrote {out}\n        {npz}")


def _layer_from_hook(hook_name: str):
    import re
    m = re.search(r"blocks\.(\d+)\.", hook_name or "")
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    main()
