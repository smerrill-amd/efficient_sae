"""Op-bucket breakdown of a BatchTopK SAE step as model size scales, plus a dump of
exactly which kernels land in the 'other' bucket."""
from __future__ import annotations
import sys, collections
sys.path.insert(0, "/wekafs/smerrill/efficient_sae")
import numpy as np, torch
import architectures  # noqa
from architectures import BatchTopKFP8TrainingSAEConfig
from architectures.fp8_formats import get_format
from sae_lens import BatchTopKTrainingSAEConfig
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainStepInput
from torch.profiler import profile, ProfilerActivity

DEV = "cuda"; B, K = 4096, 64

_CATS = [
    ("matmul/GEMM", ["addmm", "_scaled_mm", "::mm", "matmul", "bmm", "gemm",
                     "cijk", "tensile", "hipblaslt", "rocblas"]),
    ("topk/sort", ["topk", "sort", "scatter", "gather", "nonzero",
                   "rocprim", "trampoline", "cumsum", "radix", "scan", "partition"]),
    ("norm", ["norm", "frobenius"]),
    ("reduce", ["reduce", "::sum", "aten::sum", "mean", "amax", "::max", "::min"]),
    ("elementwise", ["mul", "add", "sub", "div", "relu", "clamp", "copy", "memcpy", "::to",
                     "convert", "sign", "round", "floor", "log2", "exp2", "abs", "where",
                     "neg", "fill", "zero", "pow", "sqrt", "rsqrt"]),
]
BUCKETS = ["matmul/GEMM", "topk/sort", "norm", "reduce", "elementwise", "other"]

def bucket(name):
    n = name.lower()
    for cat, keys in _CATS:
        if any(k in n for k in keys):
            return cat
    return "other"

def profile_step(cfg, autocast, d_in, d_sae, iters=10, dump_other=False):
    cls, _ = get_sae_training_class(cfg.architecture()); sae = cls(cfg).to(DEV)
    x = torch.randn(B, d_in, device=DEV, dtype=torch.bfloat16)
    def step():
        si = TrainStepInput(sae_in=x, coefficients={}, dead_neuron_mask=None,
                            n_training_steps=0, is_logging_step=False)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
        with ctx: out = sae.training_forward_pass(si)
        out.loss.backward(); sae.zero_grad(set_to_none=True)
    for _ in range(4): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): step()
        torch.cuda.synchronize()
    agg = collections.defaultdict(float); other = collections.defaultdict(float)
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", 0) or 0
        if dt <= 0: continue
        b = bucket(e.key); agg[b] += dt
        if b == "other": other[e.key] += dt
    del sae; torch.cuda.empty_cache()
    res = {b: agg.get(b, 0.0) / iters / 1e3 for b in BUCKETS}
    if dump_other:
        print("   'other' bucket contents (ms/iter):")
        for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:12]:
            print(f"      {v/iters/1e3:7.4f}  {k[:75]}")
    return res

def fp16(d_in, d_sae): return BatchTopKTrainingSAEConfig(dtype="float32", d_in=d_in, d_sae=d_sae, k=K, device=DEV, normalize_activations="none")
def fp8hw(d_in, d_sae): return BatchTopKFP8TrainingSAEConfig(dtype="bfloat16", fp8_format="e4m3", fp8_backend="hardware", d_in=d_in, d_sae=d_sae, k=K, device=DEV, normalize_activations="none")

if __name__ == "__main__":
    print("=== what's in 'other'? (FP16, d_in=1024 d_sae=8192) ===")
    profile_step(fp16(1024, 8192), True, 1024, 8192, dump_other=True)
    print("\n=== breakdown (% of step) as d_sae scales, d_in=4096, FP16 vs FP8-hw ===")
    for d_in, d_sae in [(1024, 8192), (4096, 8192), (4096, 16384), (4096, 32768), (4096, 65536), (8192, 65536)]:
        for tag, cfg, ac in [("FP16", fp16(d_in, d_sae), True), ("FP8hw", fp8hw(d_in, d_sae), False)]:
            r = profile_step(cfg, ac, d_in, d_sae)
            tot = sum(r.values())
            pct = {b: 100*r[b]/tot for b in BUCKETS}
            print(f"  d_in={d_in:5d} d_sae={d_sae:6d} {tag:6s} tot={tot:7.2f}ms | "
                  f"mm {pct['matmul/GEMM']:4.1f}% topk {pct['topk/sort']:4.1f}% "
                  f"elt {pct['elementwise']:4.1f}% red {pct['reduce']:4.1f}% "
                  f"norm {pct['norm']:4.1f}% other {pct['other']:4.1f}%")
