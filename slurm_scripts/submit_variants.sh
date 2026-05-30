#!/usr/bin/env bash
# submit_variants.sh — submit one sbatch job per model × arch combination.
#
# Hyperparameters sourced from published configs:
#   LlamaScope (arxiv 2410.20526): https://huggingface.co/fnlp/Llama-Scope
#   GemmaScope  (arxiv 2408.05147): https://huggingface.co/google/gemma-scope
#   SAELens docs: https://decoderesearch.github.io/SAELens/latest/training_saes/
#
# Usage:
#   ./submit_variants.sh                    # submit all 8 combos, mid-layer, 200M tokens
#   LAYER=16 TOKENS=500000000 ./submit_variants.sh
#   MODELS="1 3" ARCHS="topk" ./submit_variants.sh   # subset
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${HERE}/train_FP16.slurm"

LAYER="${LAYER:-}"        # empty = use mid-layer for each model
TOKENS="${TOKENS:-200000000}"
WANDB_PROJECT="${WANDB_PROJECT:-efficient_sae}"
MODELS="${MODELS:-1 2 3 4}"   # space-separated model indices
ARCHS="${ARCHS:-topk relu}"

# ── Model presets ─────────────────────────────────────────────────────────────
# Format: MODEL  DATASET  DS_CFG  D_IN  N_LAYERS  D_SAE  LR_TOPK  LR_WARM_TOPK  K  LR_RELU  LR_WARM_RELU  L1
declare -A MODEL_DEF
#                                                                                     topk params          relu params
MODEL_DEF[1]="meta-llama/Llama-3.1-8B  HuggingFaceFW/fineweb-edu  sample-10BT  4096  32  32768  8e-4  10000  50  5e-5  0     5.0"
MODEL_DEF[2]="meta-llama/Llama-3.1-70B HuggingFaceFW/fineweb-edu  sample-10BT  8192  80  65536  8e-4  10000  50  5e-5  0     5.0"
MODEL_DEF[3]="Qwen/Qwen3-8B            HuggingFaceFW/fineweb-edu  sample-10BT  4096  36  32768  8e-4  10000  50  5e-5  0     5.0"
MODEL_DEF[4]="google/gemma-3-4b        allenai/c4                 en           2560  34  16384  7e-5  1000   50  7e-5  1000  5.0"

echo "Submitting SAE training variants..."
echo "  models : ${MODELS}"
echo "  archs  : ${ARCHS}"
echo "  tokens : ${TOKENS}"
echo ""

submitted=0

for midx in ${MODELS}; do
  read -r MODEL DATASET DS_CFG D_IN N_LAYERS D_SAE \
    LR_TOPK LR_WARM_TOPK K \
    LR_RELU LR_WARM_RELU L1 <<< "${MODEL_DEF[$midx]}"

  MID=$(( N_LAYERS / 2 ))
  LAYER_USE="${LAYER:-$MID}"
  MODEL_SHORT="${MODEL##*/}"

  for ARCH in ${ARCHS}; do
    if [[ "${ARCH}" == "topk" ]]; then
      LR="${LR_TOPK}"; LR_WARM="${LR_WARM_TOPK}"
      ARCH_LABEL="topk-k${K}"
    else
      LR="${LR_RELU}"; LR_WARM="${LR_WARM_RELU}"
      ARCH_LABEL="relu-l1${L1}"
    fi

    JOB_NAME="${MODEL_SHORT}__L${LAYER_USE}__${ARCH_LABEL}"

    JOB_ID=$(sbatch \
      --job-name="${JOB_NAME}" \
      --export=ALL,\
MODEL="${MODEL}",\
DATASET="${DATASET}",\
DS_CFG="${DS_CFG}",\
D_IN="${D_IN}",\
D_SAE="${D_SAE}",\
ARCH="${ARCH}",\
LAYER="${LAYER_USE}",\
TOKENS="${TOKENS}",\
LR="${LR}",\
LR_WARM="${LR_WARM}",\
K="${K}",\
L1="${L1}",\
WANDB_PROJECT="${WANDB_PROJECT}" \
      "${SLURM_SCRIPT}" \
      | awk '{print $NF}')

    echo "  Submitted ${JOB_NAME}  →  job ${JOB_ID}"
    (( submitted++ ))
  done
done

echo ""
echo "Total jobs submitted: ${submitted}"
echo "Monitor: squeue --me"
echo "Logs:    /home/smerrill@amd.com/efficient_sae/logs/slurm/"
