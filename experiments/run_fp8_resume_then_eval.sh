#!/usr/bin/env bash
# run_fp8_resume_then_eval.sh — orchestration:
#   1) wait for the in-flight FP16 eval sweeps to finish (frees both GPUs),
#   2) RESUME the two interrupted FP8 k-sweeps from their 211707 checkpoints
#      (explicit paths — NOT `auto`, which would grab the older 500M-token run),
#   3) eval each (PHASE=all -> train-resume then core+sparse_probing on final ckpt).
# pythia runs on GPU 0, gemma on GPU 1, in parallel. Config matches the 211707
# FP16/FP8 runs so resume lands in the SAME checkpoint hash.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TS=$(date +%Y%m%d_%H%M%S)
DATA=/wekafs/smerrill/data/pile-uncopyrighted/train/00.jsonl.zst
PY_CKPT=results/saebench_pythia_fp8/checkpoints/2cd5cda3/60000256
GM_CKPT=results/saebench_gemma_fp8/checkpoints/bf2ac401/20000768

for p in "$DATA" "$PY_CKPT/activations_store_state.safetensors" "$GM_CKPT/activations_store_state.safetensors"; do
  [[ -e "$p" ]] || { echo "[orch] MISSING required path: $p" >&2; exit 1; }
done

echo "[orch $(date)] waiting for FP16 eval sweeps (sweep_k.sh) to finish..."
while pgrep -f 'sweep_k.sh' >/dev/null 2>&1; do sleep 60; done
echo "[orch $(date)] FP16 eval finished — GPUs free. Starting FP8 resume+eval."

COMMON=( WIDTH=65536 KS=20,40,80,160,320,640 PRECISIONS=fp8 PHASE=all
         EVALS=core,sparse_probing CHECKPOINTS=final
         TRAINING_TOKENS=100000000 LOCAL_DATA="$DATA"
         DTYPE=bfloat16 SAE_DTYPE=float32 N_CHECKPOINTS=4 )

env "${COMMON[@]}" MODELS=pythia GPU=0 RESUME="$PWD/$PY_CKPT" \
  ./sweep_k.sh > "results/fp8_resume_pythia_${TS}.log" 2>&1 &
PP=$!
env "${COMMON[@]}" MODELS=gemma GPU=1 RESUME="$PWD/$GM_CKPT" \
  ./sweep_k.sh > "results/fp8_resume_gemma_${TS}.log" 2>&1 &
PG=$!
echo "[orch] launched: pythia pid=$PP (GPU0) gemma pid=$PG (GPU1)  ts=$TS"

wait "$PP"; echo "[orch $(date)] pythia fp8 resume+eval rc=$?"
wait "$PG"; echo "[orch $(date)] gemma fp8 resume+eval rc=$?"
echo "[orch $(date)] ALL FP8 resume+eval complete. Analyze with notebooks/ksweep_frontier.ipynb"
