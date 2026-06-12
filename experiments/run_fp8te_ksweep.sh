#!/usr/bin/env bash
# run_fp8te_ksweep.sh — TransformerEngine FP8 k-sweep, ONE model per GPU, in parallel.
#
#   pythia-160m -> GPU 0      gemma-2-2b -> GPU 1
#
# Each model trains a shared-model BatchTopK-TE-FP8 run over the k grid at width 65536
# (encoder/decoder are transformer_engine te.Linear under te.fp8_autocast — hybrid recipe,
# delayed scaling by default; see architectures/te_batchtopk_fp8_sae.py), then evals the
# FINAL checkpoint of every k with the credential-free SAEBench suite (core +
# sparse_probing). Compare against the published authors' BatchTopK frontier with
# notebooks/ksweep_frontier.ipynb.
#
# Overrides (env): KS, TRAINING_TOKENS, WIDTH, N_CHECKPOINTS, EVALS, FP8_SCALING, FP8_RECIPE.
#   e.g. convergence looks off?  ->  FP8_SCALING=current ./run_fp8te_ksweep.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TS=$(date +%Y%m%d_%H%M%S)
DATA="${LOCAL_DATA:-/wekafs/smerrill/data/pile-uncopyrighted/train/00.jsonl.zst}"
[[ -e "$DATA" ]] || { echo "[orch] MISSING required data path: $DATA" >&2; exit 1; }

# Disconnect-proofing: the dataset is read from the LOCAL .jsonl.zst above (no HF
# streaming), and the base models (gemma-2-2b, pythia-160m-deduped) are already in the
# HF cache, so force fully-offline HF to guarantee a dropped HF connection can't stall
# or kill the run. Set HF_OFFLINE=0 to re-enable network (e.g. first-time model pull).
if [[ "${HF_OFFLINE:-1}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  echo "[orch] HF offline mode ON (cached models + local dataset only)"
fi

COMMON=(
  WIDTH="${WIDTH:-65536}"
  KS="${KS:-20,40,160,640}"
  PRECISIONS=fp8te
  PHASE="${PHASE:-all}"                      # train then eval
  EVALS="${EVALS:-core,sparse_probing}"
  CHECKPOINTS="${CHECKPOINTS:-final}"        # eval the final ckpt -> frontier points
  TRAINING_TOKENS="${TRAINING_TOKENS:-100000000}"
  N_CHECKPOINTS="${N_CHECKPOINTS:-4}"        # intermediate ckpts saved during training
  LOCAL_DATA="$DATA"
  DTYPE=bfloat16
  SAE_DTYPE=float32
)
# Optional TE recipe/scaling overrides (empty -> train_saebench_replication.py defaults:
# recipe=hybrid, scaling=delayed, amax_history=16/max, margin=0, aux_loss=on).
[[ -n "${FP8_SCALING:-}" ]] && COMMON+=( FP8_SCALING="$FP8_SCALING" )
[[ -n "${FP8_RECIPE:-}"  ]] && COMMON+=( FP8_RECIPE="$FP8_RECIPE" )

echo "[orch $(date)] FP8-TE k-sweep  ts=$TS"
echo "[orch] ${COMMON[*]}"

# transformer_engine runs its fp8 GEMM on the process's *current* CUDA device, not the
# tensor's device, so a process that can see both GPUs crashes with a cuda:0/cuda:1
# device mismatch. Give each model a SINGLE visible physical GPU (-> it is cuda:0 inside
# the process) and pass --gpu 0. PYTHIA_PHYS/GEMMA_PHYS pick the physical cards.
PYTHIA_PHYS="${PYTHIA_PHYS:-0}"
GEMMA_PHYS="${GEMMA_PHYS:-1}"
env "${COMMON[@]}" MODELS=pythia GPU=0 \
    CUDA_VISIBLE_DEVICES="$PYTHIA_PHYS" HIP_VISIBLE_DEVICES="$PYTHIA_PHYS" \
    ./sweep_k.sh > "results/fp8te_ksweep_pythia_${TS}.log" 2>&1 &
PP=$!
env "${COMMON[@]}" MODELS=gemma GPU=0 \
    CUDA_VISIBLE_DEVICES="$GEMMA_PHYS" HIP_VISIBLE_DEVICES="$GEMMA_PHYS" \
    ./sweep_k.sh > "results/fp8te_ksweep_gemma_${TS}.log" 2>&1 &
PG=$!
echo "[orch] launched: pythia pid=$PP (phys GPU $PYTHIA_PHYS)  gemma pid=$PG (phys GPU $GEMMA_PHYS)"
printf '%s\n' "results/fp8te_ksweep_pythia_${TS}.log" "results/fp8te_ksweep_gemma_${TS}.log" \
  > /tmp/fp8te_ksweep_logs.txt
echo "[orch] log paths saved to /tmp/fp8te_ksweep_logs.txt"

wait "$PP"; echo "[orch $(date)] pythia fp8te rc=$?"
wait "$PG"; echo "[orch $(date)] gemma  fp8te rc=$?"
echo "[orch $(date)] ALL FP8-TE k-sweep + eval complete. Analyze: notebooks/ksweep_frontier.ipynb"
