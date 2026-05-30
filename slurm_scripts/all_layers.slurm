#!/bin/bash
# ============================================================================
# all_layers.slurm — SLURM job array: one task per layer, one GPU per task.
#
# Submits N_LAYERS jobs in parallel, each training an SAE on one layer.
# GPU assignment is round-robin across the 8 GPUs on the node.
#
# Hyperparameters sourced from:
#   LlamaScope (arxiv 2410.20526): https://huggingface.co/fnlp/Llama-Scope
#   GemmaScope  (arxiv 2408.05147): https://huggingface.co/google/gemma-scope
#   SAELens docs: https://decoderesearch.github.io/SAELens/latest/training_saes/
#
# Usage (submit with sbatch --export):
#   sbatch --array=0-31 --export=ALL,MODEL_IDX=1,ARCH=topk,TOKENS=200000000 all_layers.slurm
#   sbatch --array=0-33 --export=ALL,MODEL_IDX=4,ARCH=relu,TOKENS=200000000 all_layers.slurm
#
# MODEL_IDX:  1=Llama-3.1-8B  2=Llama-3.1-70B  3=Qwen3-8B  4=Gemma-3-4b
# ============================================================================
#SBATCH --partition=AIG_Models
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=sae_all_layers
#SBATCH --output=/home/smerrill@amd.com/efficient_sae/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/smerrill@amd.com/efficient_sae/logs/slurm/%x_%A_%a.err

set -euo pipefail

IMAGE="rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1"
SCRIPT="/home/smerrill@amd.com/efficient_sae/src/train_sae_FP16.py"
OUTPUT_DIR="/home/smerrill@amd.com/efficient_sae/trained_models"

# ── Model presets ─────────────────────────────────────────────────────────────
MODEL_IDX="${MODEL_IDX:-1}"
case "${MODEL_IDX}" in
  # LlamaScope (arxiv 2410.20526) — TopK k=50, lr=8e-4, warmup=10K
  1) MODEL="meta-llama/Llama-3.1-8B";  DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=4096;  D_SAE=32768;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50; LR_RELU=5e-5; LR_WARM_RELU=0;    L1=5.0 ;;
  # LlamaScope (arxiv 2410.20526) — same config, larger d_in
  2) MODEL="meta-llama/Llama-3.1-70B"; DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=8192;  D_SAE=65536;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50; LR_RELU=5e-5; LR_WARM_RELU=0;    L1=5.0 ;;
  # LlamaScope-style (no published Qwen3 SAE suite yet)
  3) MODEL="Qwen/Qwen3-8B";            DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=4096;  D_SAE=32768;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50; LR_RELU=5e-5; LR_WARM_RELU=0;    L1=5.0 ;;
  # GemmaScope (arxiv 2408.05147) — lr=7e-5, cosine warmup 1K steps
  4) MODEL="google/gemma-3-4b";        DATASET="allenai/c4";                DS_CFG="en";           D_IN=2560;  D_SAE=16384;  LR_TOPK=7e-5; LR_WARM_TOPK=1000;  K=50; LR_RELU=7e-5; LR_WARM_RELU=1000; L1=5.0 ;;
  *) echo "Unknown MODEL_IDX=${MODEL_IDX}"; exit 1 ;;
esac

# ── Per-task settings ─────────────────────────────────────────────────────────
LAYER="${SLURM_ARRAY_TASK_ID}"
ARCH="${ARCH:-topk}"
TOKENS="${TOKENS:-200000000}"
BATCH="${BATCH:-4096}"
CTX="${CTX:-1024}"
DTYPE="${DTYPE:-bfloat16}"
WANDB_PROJECT="${WANDB_PROJECT:-efficient_sae}"

# Each SLURM task gets 1 GPU; ROCR_VISIBLE_DEVICES is set by SLURM to that GPU's index.
GPU_ENV="${ROCR_VISIBLE_DEVICES:-0}"

LR_DECAY=$(( (TOKENS / BATCH) / 5 ))

if [[ "${ARCH}" == "topk" ]]; then
  LR="${LR_TOPK}"; LR_WARM="${LR_WARM_TOPK}"
  ARCH_ARGS="--arch topk --k ${K}"
else
  LR="${LR_RELU}"; LR_WARM="${LR_WARM_RELU}"
  ARCH_ARGS="--arch relu --l1-coeff ${L1}"
fi

MODEL_SHORT="${MODEL##*/}"
CONTAINER="sae_${SLURM_ARRAY_JOB_ID}_${LAYER}_${ARCH}"

echo "======================================================="
echo "  Array job: ${SLURM_ARRAY_JOB_ID}  task: ${SLURM_ARRAY_TASK_ID}"
echo "  Node:      $(hostname)  GPU(ROCR): ${GPU_ENV}"
echo "  Model:     ${MODEL}  layer=${LAYER}"
echo "  Arch:      ${ARCH}  d_sae=${D_SAE}"
echo "  Tokens:    ${TOKENS}  lr=${LR}"
echo "======================================================="

# Start a dedicated container for this task
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
  -v /home:/home \
  -e ROCR_VISIBLE_DEVICES="${GPU_ENV}" \
  -e HIP_VISIBLE_DEVICES="${GPU_ENV}" \
  -e TOKENIZERS_PARALLELISM=false \
  "${IMAGE}" \
  sleep infinity

cleanup() { docker stop "${CONTAINER}" 2>/dev/null || true; }
trap cleanup EXIT

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

echo "Done: ${MODEL_SHORT} layer=${LAYER} arch=${ARCH}"
