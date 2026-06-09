"""Large-scale FP8-vs-FP16 end-to-end throughput benchmark for the BatchTopK SAE.

Times a full training step (encode + BatchTopK + decode + loss + backward) at
LLM-residual-stream sizes, comparing:
  * FP16 baseline  : BatchTopK SAE, dtype=float32 master + bf16 autocast (train_sae_FP16 recipe)
  * FP8 e4m3 emul  : software fake-quant fp8, bf16 master (no autocast)
  * FP8 e4m3 hw    : torch._scaled_mm fp8, bf16 master (no autocast)

Reports median ms/step (warmup + median-of-N) and speedup vs the FP16 baseline.
Per the project gotcha, FP8 uses bf16 master weights + bf16 activations (keeps the
big-k topk fast on ROCm); FP16 uses fp32 master + bf16 autocast.
"""
from __future__ import annotations
import sys, numpy as np, torch
sys.path.insert(0, "/wekafs/smerrill/efficient_sae")
import architectures  # noqa: F401  registers batchtopk_fp8
from architectures import BatchTopKFP8TrainingSAEConfig
from sae_lens import BatchTopKTrainingSAEConfig
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainStepInput

DEV = "cuda"
B, K = 4096, 64
WARMUP, ITERS = 8, 30


def make_sae(cfg):
    cls, _ = get_sae_training_class(cfg.architecture())
    return cls(cfg).to(DEV)


def time_step(cfg, autocast, optimizer_step: bool):
    """Median ms for one fwd+bwd step. If ``optimizer_step`` the weights are updated
    each iter (training-realistic: any fp8 weight cache misses every step); otherwise
    the weights are frozen (inference/eval: an fp8 weight cache can hit)."""
    sae = make_sae(cfg)
    x = torch.randn(B, cfg.d_in, device=DEV, dtype=torch.bfloat16)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4) if optimizer_step else None

    def step():
        si = TrainStepInput(sae_in=x, coefficients={}, dead_neuron_mask=None,
                            n_training_steps=0, is_logging_step=False)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = sae.training_forward_pass(si)
        out.loss.backward()
        if opt is not None:
            opt.step()
        sae.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); step(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    del sae, x, opt
    torch.cuda.empty_cache()
    return float(np.median(ts))


def base(d_in, d_sae):
    return dict(d_in=d_in, d_sae=d_sae, k=K, device=DEV, normalize_activations="none")


def fp16_cfg(d_in, d_sae):
    return BatchTopKTrainingSAEConfig(dtype="float32", **base(d_in, d_sae))

def fp8_cfg(d_in, d_sae, backend):
    return BatchTopKFP8TrainingSAEConfig(dtype="bfloat16", fp8_format="e4m3",
                                         fp8_backend=backend, **base(d_in, d_sae))

if __name__ == "__main__":
    print("torch", torch.__version__, torch.cuda.get_device_name())
    print(f"B={B}, k={K}, warmup={WARMUP}, iters={ITERS}")
    print("ms per fwd+bwd step (median). FP16=fp32 master+bf16 autocast; FP8=bf16 master.")
    print("  *_train = optimizer step each iter (weights change -> fp8 weight cache misses)")
    print("  hw_froz = frozen weights (inference/eval; fp8 weight-quant cache hits)\n")
    header = (f"{'d_in':>6} {'d_sae':>7} | {'FP16':>7} {'FP8emul':>8} {'FP8hw':>7} {'hw_froz':>8} | "
              f"{'emul x':>7} {'hw x':>6} {'froz x':>7}")
    print(header); print("-" * len(header))
    for d_in in [4096, 8192]:
        for d_sae in [8192, 16384, 32768, 65536]:
            t_fp16 = time_step(fp16_cfg(d_in, d_sae), autocast=True, optimizer_step=True)
            t_emul = time_step(fp8_cfg(d_in, d_sae, "emulated"), autocast=False, optimizer_step=True)
            t_hw = time_step(fp8_cfg(d_in, d_sae, "hardware"), autocast=False, optimizer_step=True)
            t_hwf = time_step(fp8_cfg(d_in, d_sae, "hardware"), autocast=False, optimizer_step=False)
            print(f"{d_in:6d} {d_sae:7d} | {t_fp16:7.2f} {t_emul:8.2f} {t_hw:7.2f} {t_hwf:8.2f} | "
                  f"{t_fp16/t_emul:7.2f} {t_fp16/t_hw:6.2f} {t_fp16/t_hwf:7.2f}")
