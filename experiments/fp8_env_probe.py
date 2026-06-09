"""Quick env-sensitivity probe for the fp8 GEMM-only path on a few SAE shapes.
Run under different env to compare hipBLASLt vs rocBLAS and TunableOp:
  python experiments/fp8_env_probe.py
  TORCH_BLAS_PREFER_HIPBLASLT=0 python experiments/fp8_env_probe.py
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 python experiments/fp8_env_probe.py
"""
from __future__ import annotations
import os, numpy as np, torch
DEV="cuda"; DT=torch.float8_e4m3fnuz; torch.manual_seed(0)
def cuda_time(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))
def run(M,K,N):
    a=torch.randn(M,K,device=DEV,dtype=torch.bfloat16); b=torch.randn(K,N,device=DEV,dtype=torch.bfloat16)
    fmax=torch.finfo(DT).max
    sa=(a.abs().amax()/fmax).float(); sb=(b.abs().amax()/fmax).float()
    aq=(a.float()/sa).clamp(-fmax,fmax).to(DT).contiguous()
    bq=(b.float()/sb).clamp(-fmax,fmax).to(DT); bq=bq.t().contiguous().t()
    tb=cuda_time(lambda:a@b)
    tf=cuda_time(lambda:torch._scaled_mm(aq,bq,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16))
    print(f"  M={M} K={K} N={N}: bf16 {tb:.3f}ms  fp8 {tf:.3f}ms  x{tb/tf:.2f}")
if __name__=="__main__":
    print("HIPBLASLT=",os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
          "TUNABLEOP=",os.environ.get("PYTORCH_TUNABLEOP_ENABLED"))
    for M,K,N in [(4096,4096,4096),(4096,8192,32768),(4096,65536,8192),(8192,8192,8192)]:
        run(M,K,N)
