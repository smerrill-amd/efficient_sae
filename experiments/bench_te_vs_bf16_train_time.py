"""TransformerEngine-FP8 vs BF16 SAE training-time benchmark + profiler.

Two parts (see --part):

  micro  (Part A, self-contained, no model/data):
         Times a full SAE training step (encode + BatchTopK + decode + loss +
         backward + optimizer.step) on SYNTHETIC residual-stream activations, for
         the two recipes used in real training:
           * BF16  : BatchTopK SAE, fp32 master + bf16 autocast   (the "FP16" recipe)
           * FP8-TE: BatchTopK-TE SAE, te.Linear under te.fp8_autocast (hybrid/delayed)
         Reports median ms/step, tokens/s, and the PROJECTED wall-clock to train
         1M and 100M tokens *on SAE compute alone*, plus a torch.profiler op
         breakdown ("what takes the most time") for each recipe.

         This isolates exactly the part that fp8 changes (the encoder/decoder GEMMs);
         it deliberately excludes the LLM forward pass that generates activations,
         which is precision-independent and identical for both recipes. Use the
         `realtrain` driver (bench_te_vs_bf16_realtrain.sh) for the end-to-end rate.

Writes results/bench_te_vs_bf16/micro_<model>_w<W>.json for the notebook to plot.

Usage:
  python bench_te_vs_bf16_train_time.py --part micro --model gemma --width 65536 --ks 160
  python bench_te_vs_bf16_train_time.py --part micro --model gemma --width 65536 --ks 20,160,640
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/wekafs/smerrill/efficient_sae")
import architectures  # noqa: F401,E402  registers batchtopk_fp8 / batchtopk_te_fp8
from architectures import BatchTopKTEFP8TrainingSAEConfig  # noqa: E402
from sae_lens import BatchTopKTrainingSAEConfig  # noqa: E402
from sae_lens.registry import get_sae_training_class  # noqa: E402
from sae_lens.saes.sae import TrainStepInput  # noqa: E402

DEV = "cuda"

# Per-model residual-stream width (d_in), matching train_saebench_replication.py.
D_IN = {"gemma": 2304, "pythia": 768}

# SAEBench training batch is 2048 token-activations / step.
BATCH_TOKENS = 2048


def make_sae(cfg):
    cls, _ = get_sae_training_class(cfg.architecture())
    return cls(cfg).to(DEV)


def base_cfg(d_in: int, d_sae: int, k: int) -> dict:
    # Mirror the SAEBench recipe: fp32 master weights, no activation normalization.
    return dict(d_in=d_in, d_sae=d_sae, k=k, dtype="float32",
                device=DEV, normalize_activations="none",
                apply_b_dec_to_input=True)


def bf16_cfg(d_in, d_sae, k):
    return BatchTopKTrainingSAEConfig(**base_cfg(d_in, d_sae, k))


def te_cfg(d_in, d_sae, k, recipe="hybrid", scaling="delayed"):
    return BatchTopKTEFP8TrainingSAEConfig(
        fp8_recipe=recipe, fp8_scaling=scaling, **base_cfg(d_in, d_sae, k))


def _step_fn(sae, x, opt, autocast: bool):
    def step():
        si = TrainStepInput(sae_in=x, coefficients={}, dead_neuron_mask=None,
                            n_training_steps=0, is_logging_step=False)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast("cuda", enabled=False))
        with torch.profiler.record_function("forward"):
            with ctx:
                out = sae.training_forward_pass(si)
        with torch.profiler.record_function("backward"):
            out.loss.backward()
        with torch.profiler.record_function("optimizer"):
            opt.step()
            sae.zero_grad(set_to_none=True)
    return step


def time_recipe(cfg, autocast: bool, batch: int, warmup: int, iters: int):
    """Median ms for one fwd+bwd+opt step (training-realistic: optimizer steps every
    iter, so any fp8 weight re-quant happens every step like in real training).

    The synthetic activation dtype mirrors each real recipe: BF16 feeds bf16 acts
    under bf16 autocast; FP8-TE feeds fp32 acts (master dtype, no torch autocast —
    TE manages fp8 inside the GEMM), matching train_saebench_replication.py."""
    sae = make_sae(cfg)
    x_dtype = torch.bfloat16 if autocast else torch.float32
    x = torch.randn(batch, cfg.d_in, device=DEV, dtype=x_dtype)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    step = _step_fn(sae, x, opt, autocast)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    ms = float(np.median(ts))
    out = dict(
        ms_per_step_median=ms,
        ms_per_step_p10=float(np.percentile(ts, 10)),
        ms_per_step_p90=float(np.percentile(ts, 90)),
        tokens_per_s=batch / (ms / 1e3),
        peak_mem_gb=peak_mem_gb,
    )
    del sae, x, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


def profile_recipe(cfg, autocast: bool, batch: int, topn: int = 15):
    """Run torch.profiler over a handful of steps; return top ops by self CUDA time."""
    sae = make_sae(cfg)
    x_dtype = torch.bfloat16 if autocast else torch.float32
    x = torch.randn(batch, cfg.d_in, device=DEV, dtype=x_dtype)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    step = _step_fn(sae, x, opt, autocast)

    from torch.profiler import ProfilerActivity, profile, schedule
    sched = schedule(wait=1, warmup=3, active=5, repeat=1)
    rows = []
    region_us: dict[str, float] = {}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched, record_shapes=False) as prof:
        for _ in range(1 + 3 + 5):
            step()
            prof.step()
    torch.cuda.synchronize()

    ka = prof.key_averages()
    # Top kernels by self CUDA time (what actually occupies the GPU).
    def _self_cuda_us(evt):
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(evt, attr):
                return float(getattr(evt, attr))
        return 0.0

    ranked = sorted(ka, key=_self_cuda_us, reverse=True)
    for evt in ranked[:topn]:
        cu = _self_cuda_us(evt)
        if cu <= 0:
            continue
        # Kernel names (esp. rocBLAS/rocprim) can be >1KB; keep a readable prefix.
        name = evt.key if len(evt.key) <= 110 else evt.key[:107] + "..."
        rows.append(dict(op=name, self_cuda_us=cu, count=int(evt.count)))

    # Coarse region totals (forward / backward / optimizer) via record_function labels.
    for evt in ka:
        if evt.key in ("forward", "backward", "optimizer"):
            tot = 0.0
            for attr in ("device_time_total", "cuda_time_total"):
                if hasattr(evt, attr):
                    tot = float(getattr(evt, attr))
                    break
            region_us[evt.key] = tot

    del sae, x, opt
    torch.cuda.empty_cache()
    return dict(top_ops=rows, region_cuda_us=region_us)


def project(ms_per_step: float, batch: int) -> dict:
    steps_1m = 1_000_000 / batch
    steps_100m = 100_000_000 / batch
    return dict(
        sec_for_1M=steps_1m * ms_per_step / 1e3,
        sec_for_100M=steps_100m * ms_per_step / 1e3,
        hours_for_100M=steps_100m * ms_per_step / 1e3 / 3600.0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", default="micro", choices=["micro"],
                   help="micro: synthetic SAE-step microbench (Part A).")
    p.add_argument("--model", default="gemma", choices=list(D_IN))
    p.add_argument("--width", type=int, default=65536, help="d_sae (dictionary width).")
    p.add_argument("--ks", default="160", help="Comma list of k values to benchmark.")
    p.add_argument("--batch", type=int, default=BATCH_TOKENS,
                   help="Token-activations per step (SAEBench=2048).")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--recipe", default="hybrid", choices=["hybrid", "e4m3", "e5m2"])
    p.add_argument("--scaling", default="delayed", choices=["delayed", "current"])
    p.add_argument("--out-dir", default="/wekafs/smerrill/efficient_sae/experiments/"
                                        "results/bench_te_vs_bf16")
    args = p.parse_args()

    d_in = D_IN[args.model]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    print(f"model={args.model} d_in={d_in} width={args.width} ks={ks} "
          f"batch={args.batch} (warmup={args.warmup}, iters={args.iters})")
    print(f"BF16 = fp32 master + bf16 autocast   |   FP8-TE = te.Linear "
          f"({args.recipe}/{args.scaling})")
    print("=" * 78)

    results = dict(
        meta=dict(
            torch=torch.__version__, device=torch.cuda.get_device_name(),
            model=args.model, d_in=d_in, width=args.width, batch=args.batch,
            warmup=args.warmup, iters=args.iters,
            recipe=args.recipe, scaling=args.scaling,
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
        ),
        by_k={},
    )

    hdr = (f"{'k':>5} | {'BF16 ms':>8} {'FP8TE ms':>9} {'speedup':>8} | "
           f"{'BF16 h/100M':>12} {'FP8TE h/100M':>13} | {'BF16 GB':>8} {'FP8TE GB':>9}")
    print(hdr)
    print("-" * len(hdr))

    for k in ks:
        bf16 = time_recipe(bf16_cfg(d_in, args.width, k), autocast=True,
                           batch=args.batch, warmup=args.warmup, iters=args.iters)
        te = time_recipe(te_cfg(d_in, args.width, k, args.recipe, args.scaling),
                         autocast=False, batch=args.batch,
                         warmup=args.warmup, iters=args.iters)
        bf16_proj = project(bf16["ms_per_step_median"], args.batch)
        te_proj = project(te["ms_per_step_median"], args.batch)
        speedup = bf16["ms_per_step_median"] / te["ms_per_step_median"]

        bf16_prof = profile_recipe(bf16_cfg(d_in, args.width, k), autocast=True,
                                   batch=args.batch)
        te_prof = profile_recipe(te_cfg(d_in, args.width, k, args.recipe, args.scaling),
                                 autocast=False, batch=args.batch)

        results["by_k"][str(k)] = dict(
            bf16=dict(**bf16, projection=bf16_proj, profile=bf16_prof),
            fp8te=dict(**te, projection=te_proj, profile=te_prof),
            speedup_bf16_over_fp8te=speedup,
        )
        print(f"{k:5d} | {bf16['ms_per_step_median']:8.3f} "
              f"{te['ms_per_step_median']:9.3f} {speedup:7.2f}x | "
              f"{bf16_proj['hours_for_100M']:12.3f} {te_proj['hours_for_100M']:13.3f} | "
              f"{bf16['peak_mem_gb']:8.2f} {te['peak_mem_gb']:9.2f}")

    out_path = out_dir / f"micro_{args.model}_w{args.width}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("=" * 78)
    print(f"wrote {out_path}")

    # Brief profiler readout for the smallest k (illustrative).
    k0 = str(ks[0])
    for name in ("bf16", "fp8te"):
        prof = results["by_k"][k0][name]["profile"]
        print(f"\n[{name}] top GPU ops (k={k0}, self CUDA us):")
        for r in prof["top_ops"][:8]:
            print(f"  {r['self_cuda_us']:12.1f}us  x{r['count']:<5d}  {r['op']}")


if __name__ == "__main__":
    main()
