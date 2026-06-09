"""Deep-dive microbenchmark: why is torch._scaled_mm (FP8 hardware GEMM) slow for
the SAE matmul shapes on this ROCm / MI300X box?

We isolate the FP8 hardware GEMM (``torch._scaled_mm``) on exactly the shapes the
BatchTopK SAE uses and compare it against a plain bf16 matmul:

    encode:  [B, d_in] @ [d_in, d_sae]
    decode:  [B, d_sae] @ [d_sae, d_in]

and probe the usual suspects for bad ROCm fp8 perf:
  * proper warmup + CUDA-event timing, median of N
  * per-tensor vs per-row (rowwise) scaling
  * the cost of the quantize/cast/contiguous prologue vs the GEMM itself
  * hipBLASLt vs rocBLAS path and TunableOp autotuning (toggled via env, see __main__)

Run variants (set BEFORE launching python):
  baseline:            python experiments/fp8_scaled_mm_deepdive.py
  force hipBLASLt:     TORCH_BLAS_PREFER_HIPBLASLT=1 python ...
  disable hipBLASLt:   TORCH_BLAS_PREFER_HIPBLASLT=0 python ...
  tunableop:           PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 python ...
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

DEV = "cuda"
DT = torch.float8_e4m3fnuz          # MI300 native fp8
torch.manual_seed(0)


def cuda_time(fn, warmup=10, iters=50):
    """Median ms/call via CUDA events, after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def make_fp8_operands(M, K, N, rowwise=False):
    """Return (aq, bq, sa, sb) ready for torch._scaled_mm.

    a:[M,K] row-major; b stored column-major (b.t().contiguous().t()).
    per-tensor scales are scalars; rowwise: sa=[M,1], sb=[1,N].
    """
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    fmax = torch.finfo(DT).max
    if rowwise:
        a_amax = a.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).float()  # [M,1]
        b_amax = b.abs().amax(dim=0, keepdim=True).clamp(min=1e-12).float()  # [1,N]
        sa = a_amax / fmax
        sb = b_amax / fmax
        aq = (a.float() / sa).clamp(-fmax, fmax).to(DT).contiguous()
        bq = (b.float() / sb).clamp(-fmax, fmax).to(DT)
        bq = bq.t().contiguous().t()
        # _scaled_mm rowwise wants scale_a [M,1], scale_b [1,N]
        return aq, bq, sa, sb
    a_amax = a.abs().amax().clamp(min=1e-12).float()
    b_amax = b.abs().amax().clamp(min=1e-12).float()
    sa = a_amax / fmax
    sb = b_amax / fmax
    aq = (a.float() / sa).clamp(-fmax, fmax).to(DT).contiguous()
    bq = (b.float() / sb).clamp(-fmax, fmax).to(DT)
    bq = bq.t().contiguous().t()
    return aq, bq, sa, sb


def bench_shape(M, K, N):
    """Compare bf16 mm vs fp8 _scaled_mm (GEMM-only and full incl. quantize) for [M,K]@[K,N]."""
    a16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b16 = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)

    t_bf16 = cuda_time(lambda: a16 @ b16)

    # --- per-tensor fp8, GEMM only (operands pre-quantized) ---
    aq, bq, sa, sb = make_fp8_operands(M, K, N, rowwise=False)
    def gemm_pt():
        return torch._scaled_mm(aq, bq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    try:
        t_fp8_gemm = cuda_time(gemm_pt)
    except Exception as ex:
        t_fp8_gemm = float("nan"); print("   per-tensor gemm FAILED:", repr(ex)[:120])

    # --- per-row fp8, GEMM only ---
    try:
        aqr, bqr, sar, sbr = make_fp8_operands(M, K, N, rowwise=True)
        def gemm_rw():
            return torch._scaled_mm(aqr, bqr, scale_a=sar, scale_b=sbr, out_dtype=torch.bfloat16)
        t_fp8_rw = cuda_time(gemm_rw)
    except Exception as ex:
        t_fp8_rw = float("nan"); print("   rowwise gemm FAILED:", repr(ex)[:120])

    # --- full fp8 incl. dynamic quantize prologue (what fp8_ops actually does) ---
    fmax = torch.finfo(DT).max
    def full_pt():
        a_amax = a16.detach().abs().amax().clamp(min=1e-12).float()
        b_amax = b16.detach().abs().amax().clamp(min=1e-12).float()
        s_a = a_amax / fmax; s_b = b_amax / fmax
        x = (a16.float() / s_a).clamp(-fmax, fmax).to(DT).contiguous()
        w = (b16.float() / s_b).clamp(-fmax, fmax).to(DT)
        w = w.t().contiguous().t()
        return torch._scaled_mm(x, w, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    try:
        t_full = cuda_time(full_pt)
    except Exception as ex:
        t_full = float("nan"); print("   full fp8 FAILED:", repr(ex)[:120])

    flops = 2 * M * K * N
    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12 if ms == ms and ms > 0 else float("nan")

    print(f"  M={M:5d} K={K:5d} N={N:6d} | "
          f"bf16 {t_bf16:7.3f}ms ({tflops(t_bf16):5.0f} TF) | "
          f"fp8-gemm-pt {t_fp8_gemm:7.3f}ms ({tflops(t_fp8_gemm):5.0f} TF) "
          f"x{t_bf16/t_fp8_gemm:4.2f} | "
          f"fp8-gemm-rw {t_fp8_rw:7.3f}ms x{t_bf16/t_fp8_rw:4.2f} | "
          f"fp8-full {t_full:7.3f}ms x{t_bf16/t_full:4.2f}")
    return dict(M=M, K=K, N=N, bf16=t_bf16, fp8_gemm_pt=t_fp8_gemm,
                fp8_gemm_rw=t_fp8_rw, fp8_full=t_full)


if __name__ == "__main__":
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name())
    print("env: TORCH_BLAS_PREFER_HIPBLASLT=", os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
          " PYTORCH_TUNABLEOP_ENABLED=", os.environ.get("PYTORCH_TUNABLEOP_ENABLED"))
    B = 4096
    print("\n=== SAE ENCODE shapes:  [B, d_in] @ [d_in, d_sae] ===")
    for d_in in [1024, 2048, 4096, 8192]:
        for d_sae in [8192, 16384, 32768, 65536]:
            bench_shape(B, d_in, d_sae)
    print("\n=== SAE DECODE shapes:  [B, d_sae] @ [d_sae, d_in] ===")
    for d_sae in [8192, 16384, 32768, 65536]:
        for d_in in [1024, 2048, 4096, 8192]:
            bench_shape(B, d_sae, d_in)
    print("\n=== square / known-anomaly shapes ===")
    for n in [2048, 4096, 8192]:
        bench_shape(n, n, n)
