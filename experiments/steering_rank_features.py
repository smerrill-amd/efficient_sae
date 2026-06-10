#!/usr/bin/env python3
"""steering_rank_features.py — re-rank concrete features OFFLINE (no GPU / no model).

steering_find_features.py does the one expensive pass over the model and dumps the
raw per-feature material (top-activating tokens, firing counts, max-context windows)
to ``concrete_features.npz``. This script re-ranks that material under different
selection criteria instantly, so we can iterate on *which* features to steer without
ever re-running gemma.

Two flavours of "concrete":
  --mode single   single-literal-token detectors: share ~1.0 (fires on ONE token,
                  e.g. 'amssymb'). Cleanest causal handle, but often esoteric.
  --mode class    semantic token-CLASS features: share in [share-min, share-max] with
                  low entropy — fires on a small RELATED set (e.g. ' France'/' Germany'
                  /' Spain' -> a countries feature). The richer "concept" features.

Example:
  python steering_rank_features.py \
      --npz results/saebench_gemma_fp8/w65536_k80/steering/concrete_features.npz \
      --mode class --min-freq 5e-4 --top 30
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "for", "on", "at", "by", "with", "as", "that", "this",
    "it", "its", "from", "but", "not", "no", "so", "if", "then", "than", "i",
    "you", "he", "she", "they", "we", "his", "her", "their", "our", "my",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--npz", type=Path, required=True,
                   help="concrete_features.npz written by steering_find_features.py.")
    p.add_argument("--json", type=Path, default=None,
                   help="Companion concrete_features.json (for model_name; default: "
                        "sibling of --npz).")
    p.add_argument("--mode", choices=["single", "class"], default="class",
                   help="single = literal-token detectors (share~1); "
                        "class = semantic token-class concept features (default).")
    p.add_argument("--min-freq", type=float, default=2e-4)
    p.add_argument("--max-freq", type=float, default=0.1)
    p.add_argument("--share-min", type=float, default=None,
                   help="Min top-token share (default: 0.9 for single, 0.15 for class).")
    p.add_argument("--share-max", type=float, default=None,
                   help="Max top-token share (default: 1.0 for single, 0.85 for class).")
    p.add_argument("--content-only", dest="content_only", action="store_true",
                   default=True, help="Require the dominant token to be a content word.")
    p.add_argument("--allow-structural", dest="content_only", action="store_false")
    p.add_argument("--avoid-markup", action="store_true", default=False,
                   help="Drop features whose context looks like LaTeX/markup "
                        "(contains a backslash or '{').")
    p.add_argument("--top", type=int, default=30, help="How many to print.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON of the selected features (default: print only).")
    return p.parse_args()


def is_content(s: str) -> bool:
    s = s.strip()
    return bool(s) and any(c.isalpha() for c in s) and s.lower() not in STOPWORDS


def main() -> None:
    args = parse_args()
    import numpy as np

    json_path = args.json or args.npz.with_suffix(".json")
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    model_name = meta.get("model_name", "google/gemma-2-2b")

    # Tokenizer only (no model weights) to decode token ids.
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    d = np.load(args.npz)
    share = d["top_token_share"]
    freq = d["firing_freq"]
    firing = d["firing"]
    maxact = d["max_activation"]
    top_toks = d["top_toks"]            # [F, N]
    top_vals = d["top_vals"]            # [F, N]
    glob_win = d["glob_win"]            # [F, 2w+1]
    F, N = top_toks.shape

    share_min = args.share_min if args.share_min is not None else (
        0.9 if args.mode == "single" else 0.15)
    share_max = args.share_max if args.share_max is not None else (
        1.0 if args.mode == "single" else 0.85)

    valid = top_vals > (np.finfo(np.float32).min / 2)
    keep = ((freq >= args.min_freq) & (freq <= args.max_freq)
            & (firing >= N) & (share >= share_min) & (share <= share_max))

    # Rank: 'single' by share (peakedness); 'class' by a concept score that rewards a
    # small-but-not-singleton token set firing strongly and often.
    order = np.argsort(-(share + 1e-6 * np.maximum(maxact, 0)))
    if args.mode == "class":
        # entropy proxy from share: want low entropy yet share<1 -> a tight class.
        concept = (1.0 - np.abs(share - 0.4)) + 1e-6 * np.maximum(maxact, 0)
        order = np.argsort(-concept)

    print(f"  mode={args.mode}  share in [{share_min},{share_max}]  "
          f"freq in [{args.min_freq},{args.max_freq}]  ({int(keep.sum())} pass filters)\n")
    print(f"  {'feat':>6}  {'share':>5}  {'freq':>8}  {'maxact':>7}  "
          f"top tokens (the 'class')   |  max context")

    selected = []
    for f in order.tolist():
        if not keep[f]:
            continue
        f_valid = valid[f]
        toks_f = top_toks[f][f_valid]
        if toks_f.size == 0:
            continue
        uniq, counts = np.unique(toks_f, return_counts=True)
        dom_id = int(uniq[counts.argmax()])
        dom_str = tok.decode([dom_id])
        if args.content_only and not is_content(dom_str):
            continue
        ctx = tok.decode(glob_win[f].tolist())
        if args.avoid_markup and ("\\" in ctx or "{" in ctx):
            continue
        # The token "class": most common distinct tokens among the top activations.
        top_order = np.argsort(-counts)[:6]
        klass = [tok.decode([int(uniq[i])]) for i in top_order]
        entropy = float(-(counts / counts.sum() *
                          np.log(counts / counts.sum() + 1e-12)).sum())
        selected.append({
            "feature": int(f),
            "top_token_share": round(float(share[f]), 3),
            "token_entropy": round(entropy, 3),
            "firing_freq": round(float(freq[f]), 6),
            "max_activation": round(float(maxact[f]), 2),
            "token_class": klass,
            "max_context": ctx,
        })
        kshow = " ".join(repr(t) for t in klass)[:40]
        cshow = ctx.replace("\n", " ")[:46]
        print(f"  {f:>6}  {share[f]:>5.2f}  {freq[f]:>8.5f}  {maxact[f]:>7.2f}  "
              f"{kshow:<40}  |  {cshow}")
        if len(selected) >= args.top:
            break

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"npz": str(args.npz), "model_name": model_name, "mode": args.mode,
             "selected": selected}, indent=2, ensure_ascii=False))
        print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
