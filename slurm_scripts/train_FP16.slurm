#!/bin/bash
# ============================================================================
# train.slurm — run one SAE training job inside the ROCm Docker container.
#
# Do not submit this directly; use submit_variants.sh or all_layers.slurm.
#
# Required env vars (pass with sbatch --export=ALL,VAR=val,...):
#   MODEL      — HuggingFace model name  (e.g. meta-llama/Llama-3.1-8B)
#   DATASET    — HuggingFace dataset     (e.g. HuggingFaceFW/fineweb-edu)
#   DS_CFG     — dataset subset          (e.g. sample-10BT)
#   D_IN       — hook input dimension    (e.g. 4096)
#   D_SAE      — SAE dictionary size     (e.g. 32768)
#   ARCH       — topk | relu
#   LAYER      — transformer layer index
#   TOKENS     — total training tokens   (e.g. 200000000)
#   LR         — learning rate
#   LR_WARM    — lr warmup steps
#   BATCH      — batch size in tokens    (default 4096)
#   CTX        — context size            (default 1024)
#   DTYPE      — bfloat16 | float16 | float32
#   WANDB_PROJECT — W&B project name
#   # For topk:
#   K          — number of active features
#   # For relu:
#   L1         — l1 coefficient
# ============================================================================
#SBATCH --partition=AIG_Models
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=sae_%x
# Log paths are relative to the submission directory (SLURM_SUBMIT_DIR), so submit
# from the project root (or via submit_variants.sh, which sets --chdir).
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# Project root: prefer an explicit PROJECT_ROOT, else the dir sbatch was run from.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

IMAGE="rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/train_sae_FP16.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/trained_models}"
CONTAINER="sae_${SLURM_JOB_ID}"

# Defaults
BATCH="${BATCH:-4096}"
CTX="${CTX:-1024}"
DTYPE="${DTYPE:-bfloat16}"
WANDB_PROJECT="${WANDB_PROJECT:-efficient_sae}"
LR_DECAY=$(( (TOKENS / BATCH) / 5 ))

# SLURM sets ROCR_VISIBLE_DEVICES to the allocated GPU (e.g. "0" or "3").
# Pass it into Docker so the container only sees that one GPU.
GPU_ENV="${ROCR_VISIBLE_DEVICES:-0}"

echo "======================================================="
echo "  Job:       ${SLURM_JOB_ID}  node=$(hostname)"
echo "  Model:     ${MODEL}  layer=${LAYER}"
echo "  Arch:      ${ARCH}   d_sae=${D_SAE}"
echo "  Tokens:    ${TOKENS}  lr=${LR}  dtype=${DTYPE}"
echo "  GPU(ROCR): ${GPU_ENV}"
echo "======================================================="

# Build arch-specific args
if [[ "${ARCH}" == "topk" ]]; then
  ARCH_ARGS="--arch topk --k ${K}"
else
  ARCH_ARGS="--arch relu --l1-coeff ${L1}"
fi

# Start a dedicated container for this job (auto-removed on exit via --rm)
docker run -d \
  --rm \
  --name "${CONTAINER}" \
  --network=host \
  --ipc=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 200G \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  -e ROCR_VISIBLE_DEVICES="${GPU_ENV}" \
  -e HIP_VISIBLE_DEVICES="${GPU_ENV}" \
  -e TOKENIZERS_PARALLELISM=false \
  "${IMAGE}" \
  sleep infinity

# Ensure container is stopped even if the job is cancelled or errors out
cleanup() { docker stop "${CONTAINER}" 2>/dev/null || true; }
trap cleanup EXIT

# Run training
docker exec "${CONTAINER}" python3 "${SCRIPT}" \
  --model            "${MODEL}" \
  --dataset          "${DATASET}" \
  --dataset-config   "${DS_CFG}" \
  --hook-name        "blocks.${LAYER}.hook_resid_post" \
  --d-in             "${D_IN}" \
  --d-sae            "${D_SAE}" \
  --lr               "${LR}" \
  --lr-warm-up-steps "${LR_WARM}" \
  --lr-decay-steps   "${LR_DECAY}" \
  --lr-scheduler     "constant" \
  --batch-size       "${BATCH}" \
  --context-size     "${CTX}" \
  --training-tokens  "${TOKENS}" \
  --dtype            "${DTYPE}" \
  --device           "cuda:0" \
  --llm-device       "cuda:0" \
  --output-dir       "${OUTPUT_DIR}" \
  --wandb-project    "${WANDB_PROJECT}" \
  ${ARCH_ARGS}

echo "Training complete for layer ${LAYER} (${ARCH})"
