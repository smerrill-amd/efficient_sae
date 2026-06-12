"""Part B: end-to-end (real-trainer) steady-state throughput, FP8-TE vs BF16.

Runs train_saebench_replication.py for a short token budget on a SINGLE SAE
(one width, one k) for each recipe, parses the tqdm token-rate, and reports the
steady-state tokens/s (median of the post-warmup window) plus the projected
wall-clock to train 100M tokens END TO END (LLM forward + activation store +
SAE step — i.e. everything, not just the SAE compute the microbench isolates).

Contrast with bench_te_vs_bf16_train_time.py (Part A): the microbench shows the
SAE-step speedup in isolation; here the precision-independent LLM forward that
generates activations is included, so the realized end-to-end speedup is diluted.

Writes results/bench_te_vs_bf16/realtrain_<model>_w<W>.json for the notebook.

Usage:
  python bench_te_vs_bf16_realtrain.py --model gemma --width 65536 --k 160 \
      --tokens 2500000 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_saebench_replication.py"
DEFAULT_DATA = "/wekafs/smerrill/data/pile-uncopyrighted/train/00.jsonl.zst"

RATE_RE = re.compile(r"([\d.]+)it/s")


def run_one(recipe: str, model: str, width: int, k: int, tokens: int,
            gpu: int, data: str, log_path: Path) -> dict:
    """Run the trainer once for `recipe` in {bf16, fp8te}; return throughput stats."""
    import os

    cmd = [
        "python3", str(TRAIN),
        "--model", model,
        "--gpu", "0",                 # single visible GPU -> it's cuda:0 in-process
        "--widths", str(width),
        "--ks", str(k),
        "--training-tokens", str(tokens),
        "--n-checkpoints", "0",
        "--no-save-final",
        "--no-wandb",
        "--local-data", data,
        "--dtype", "bfloat16",
        "--sae-dtype", "float32",
    ]
    if recipe == "fp8te":
        cmd.append("--fp8-te")
    elif recipe != "bf16":
        raise ValueError(recipe)

    env = dict(os.environ)
    # TE runs its fp8 GEMM on the process's current CUDA device, so pin ONE physical
    # GPU (becomes cuda:0 in-process) for both recipes to keep the comparison fair.
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"

    print(f"\n>>> [{recipe}] {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, cwd=str(HERE), env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    text = log_path.read_text(errors="replace")
    rates = [float(m) for m in RATE_RE.findall(text)]
    rc = proc.returncode

    if not rates:
        return dict(recipe=recipe, returncode=rc, wall_s=wall,
                    error="no it/s rates parsed (run may have failed; check log)",
                    log=str(log_path))

    # Drop the first 40% (model load, first buffer fills, autotuning) -> steady state.
    n = len(rates)
    steady = rates[int(0.4 * n):] or rates
    steady_sorted = sorted(steady)
    median = steady_sorted[len(steady_sorted) // 2]
    proj_100m_h = 100_000_000 / median / 3600.0

    return dict(
        recipe=recipe, returncode=rc, wall_s=wall,
        n_rate_samples=n,
        tokens_per_s_steady_median=median,
        tokens_per_s_steady_mean=sum(steady) / len(steady),
        tokens_per_s_min=min(rates), tokens_per_s_max=max(rates),
        hours_for_100M=proj_100m_h,
        log=str(log_path),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemma", choices=["gemma", "pythia"])
    p.add_argument("--width", type=int, default=65536)
    p.add_argument("--k", type=int, default=160)
    p.add_argument("--tokens", type=int, default=2_500_000,
                   help="Short token budget per recipe for the steady-state probe.")
    p.add_argument("--gpu", type=int, default=0, help="Physical GPU index to pin.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default=str(HERE / "results" / "bench_te_vs_bf16"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    results = dict(
        meta=dict(model=args.model, width=args.width, k=args.k, tokens=args.tokens,
                  gpu=args.gpu, timestamp=ts),
        recipes={},
    )
    for recipe in ("bf16", "fp8te"):
        log_path = out_dir / f"realtrain_{args.model}_w{args.width}_{recipe}_{ts}.log"
        stats = run_one(recipe, args.model, args.width, args.k, args.tokens,
                        args.gpu, args.data, log_path)
        results["recipes"][recipe] = stats
        print(f"    -> {recipe}: " + (stats.get("error") or
              f"{stats['tokens_per_s_steady_median']:.0f} tok/s steady, "
              f"{stats['hours_for_100M']:.2f} h/100M (rc={stats['returncode']})"))

    bf, te = results["recipes"]["bf16"], results["recipes"]["fp8te"]
    if "tokens_per_s_steady_median" in bf and "tokens_per_s_steady_median" in te:
        results["end2end_speedup_fp8te_over_bf16"] = (
            te["tokens_per_s_steady_median"] / bf["tokens_per_s_steady_median"])
        print(f"\nend-to-end speedup (fp8te/bf16): "
              f"{results['end2end_speedup_fp8te_over_bf16']:.3f}x")

    out_path = out_dir / f"realtrain_{args.model}_w{args.width}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
