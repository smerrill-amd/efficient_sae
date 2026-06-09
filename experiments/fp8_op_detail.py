"""Detailed per-op profile of one BatchTopK SAE training step, to see what the
'elementwise' bucket actually is and whether it should be faster."""
from __future__ import annotations
import sys
sys.path.insert(0, "/wekafs/smerrill/efficient_sae")
import torch
import architectures  # noqa
from architectures import BatchTopKFP8TrainingSAEConfig
from sae_lens import BatchTopKTrainingSAEConfig
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainStepInput
from torch.profiler import profile, ProfilerActivity

DEV="cuda"; B,D_IN,D_SAE,K = 4096,1024,8192,64

def run(name, cfg, autocast):
    cls,_=get_sae_training_class(cfg.architecture()); sae=cls(cfg).to(DEV)
    x=torch.randn(B,D_IN,device=DEV,dtype=torch.bfloat16)
    def step():
        si=TrainStepInput(sae_in=x,coefficients={},dead_neuron_mask=None,n_training_steps=0,is_logging_step=False)
        ctx=torch.autocast("cuda",dtype=torch.bfloat16) if autocast else torch.autocast("cuda",enabled=False)
        with ctx: out=sae.training_forward_pass(si)
        out.loss.backward(); sae.zero_grad(set_to_none=True)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        for _ in range(10): step()
        torch.cuda.synchronize()
    print(f"\n===== {name}: top 20 ops by GPU self-time (ms/iter over 10 iters) =====")
    rows=[]
    for e in prof.key_averages():
        dt=getattr(e,"self_device_time_total",0) or 0
        if dt>0: rows.append((e.key, dt/10/1e3, getattr(e,"count",0)))
    rows.sort(key=lambda r:-r[1])
    tot=sum(r[1] for r in rows)
    for k,ms,cnt in rows[:20]:
        print(f"  {ms:7.3f} ms  {100*ms/tot:5.1f}%  n={cnt:4d}  {k[:70]}")
    print(f"  total GPU self-time: {tot:.3f} ms/iter")

if __name__=="__main__":
    base=dict(d_in=D_IN,d_sae=D_SAE,k=K,device=DEV,normalize_activations="none")
    run("FP16 (bf16)", BatchTopKTrainingSAEConfig(dtype="float32",**base), True)
    run("FP8 e4m3 emul", BatchTopKFP8TrainingSAEConfig(dtype="bfloat16",fp8_format="e4m3",fp8_backend="emulated",**base), False)
