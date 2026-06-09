"""Break down the FP8 'full' path to localize the overhead, and test fixes.

The deep-dive showed _scaled_mm itself is 1.3-1.8x faster than bf16, but the
end-to-end fp8 path (quantize prologue + GEMM) is 2-4x SLOWER. This script
decomposes the prologue (amax-reduce, cast, transpose-contiguous) and compares
the current fp8_ops recipe against an optimized one:

  current:  a.float()/sa -> clamp -> .to(fp8) -> contiguous ;  b.float()/sb -> .to(fp8) -> .t().contiguous().t()
  opt_act:  quantize activation directly from bf16 (no .float roundtrip)
  amortized weight: in real training the weight changes once/step, so its fp8
                    column-major copy can be cached instead of rebuilt per call.
"""
from __future__ import annotations
import numpy as np, torch

DEV = "cuda"; DT = torch.float8_e4m3fnuz
torch.manual_seed(0)

def cuda_time(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))

def quant_pt(t, fmax):
    amax = t.detach().abs().amax().clamp(min=1e-12).float()
    s = amax / fmax
    q = (t.float() / s).clamp(-fmax, fmax).to(DT)
    return q, s

def quant_pt_nofloat(t, fmax):
    amax = t.detach().abs().amax().clamp(min=1e-12)
    s = (amax / fmax)
    q = (t / s).clamp(-fmax, fmax).to(DT)   # cast bf16->fp8 directly
    return q, s.float()

def breakdown(M, K, N):
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    fmax = torch.finfo(DT).max
    t_bf16 = cuda_time(lambda: a @ b)

    # prologue components
    t_amax = cuda_time(lambda: a.detach().abs().amax().clamp(min=1e-12).float())
    sa = (a.abs().amax()/fmax).float(); sb = (b.abs().amax()/fmax).float()
    t_cast_a = cuda_time(lambda: (a.float()/sa).clamp(-fmax,fmax).to(DT).contiguous())
    t_wtrans = cuda_time(lambda: (b.float()/sb).clamp(-fmax,fmax).to(DT).t().contiguous().t())

    # pre-quantized operands -> GEMM only
    aq,_ = quant_pt(a, fmax); bq,_ = quant_pt(b, fmax); bq = bq.t().contiguous().t()
    t_gemm = cuda_time(lambda: torch._scaled_mm(aq, bq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))

    # full current recipe
    def full_current():
        x, s_a = quant_pt(a, fmax); x = x.contiguous()
        w, s_b = quant_pt(b, fmax); w = w.t().contiguous().t()
        return torch._scaled_mm(x, w, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    t_full = cuda_time(full_current)

    # full path but quantizing in bf16 (no .float roundtrip), weight still rebuilt each call
    def full_nofloat():
        x, s_a = quant_pt_nofloat(a, fmax); x = x.contiguous()
        w, s_b = quant_pt_nofloat(b, fmax); w = w.t().contiguous().t()
        return torch._scaled_mm(x, w, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    t_nofloat = cuda_time(full_nofloat)

    # optimized: weight pre-quantized once (amortized), activation quantized w/o float roundtrip
    wq, s_b = quant_pt_nofloat(b, fmax); wq = wq.t().contiguous().t()
    def amortized():
        x, s_a = quant_pt_nofloat(a, fmax); x = x.contiguous()
        return torch._scaled_mm(x, wq, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    t_amort = cuda_time(amortized)

    print(f"  M={M:5d} K={K:6d} N={N:6d} | bf16 {t_bf16:7.3f} | gemm-only {t_gemm:7.3f}(x{t_bf16/t_gemm:.2f}) "
          f"| wTrans {t_wtrans:7.3f} | full {t_full:7.3f}(x{t_bf16/t_full:.2f}) "
          f"| nofloat {t_nofloat:7.3f}(x{t_bf16/t_nofloat:.2f}) | amort-W {t_amort:7.3f}(x{t_bf16/t_amort:.2f})")

if __name__ == "__main__":
    print("torch", torch.__version__, torch.cuda.get_device_name())
    print("ENCODE [B,d_in]@[d_in,d_sae]")
    for K,N in [(1024,8192),(4096,16384),(8192,32768),(8192,65536)]:
        breakdown(4096, K, N)
    print("DECODE [B,d_sae]@[d_sae,d_in]")
    for K,N in [(8192,1024),(16384,2048),(32768,4096),(65536,8192)]:
        breakdown(4096, K, N)
