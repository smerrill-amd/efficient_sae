#!/usr/bin/env bash
# train_interactive.sh — select a model, layer(s), and GPU; launch SAE training.
#
# Hyperparameters are sourced directly from the following published configs:
#
#  LlamaScope  — Llama-3.1-8B/70B and Qwen3-8B (TopK)
#    Paper : https://arxiv.org/abs/2410.20526
#    Repo  : https://github.com/OpenMOSS/Language-Model-SAEs
#    HF    : https://huggingface.co/fnlp/Llama-Scope
#    Params: TopK k=50, d_sae=32K (8×), lr=8e-4, warmup=10K steps,
#            linear decay last 20%, batch=4096, ctx=1024, Adam β=(0.9,0.999)
#
#  GemmaScope  — Gemma-3-4b (closest published config is Gemma-2)
#    Paper : https://arxiv.org/abs/2408.05147
#    HF    : https://huggingface.co/google/gemma-scope
#    Params: JumpReLU (approximated here as TopK), lr=7e-5, cosine warmup 1K steps,
#            batch=4096, ctx=1024, Adam β=(0,0.999), 4B–8B training tokens
#
#  SAELens training docs (general reference)
#    https://decoderesearch.github.io/SAELens/latest/training_saes/
#
set -euo pipefail

# Resolve the project root from this script's location (shell_scripts/ -> project root)
# so the script works regardless of where the repo lives. Override by exporting
# PROJECT_ROOT before invoking.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# ---------------------------------------------------------------------------
# Load secrets from .env (optional)
# ---------------------------------------------------------------------------
# W&B reads WANDB_API_KEY from the environment, so we source an optional,
# gitignored .env at the project root (no interactive `wandb login` needed).
# Best-effort: if .env is missing or empty, we just continue.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a                          # auto-export every var defined while sourcing
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Loaded W&B credentials from ${ENV_FILE}"
  fi
fi

# Runs inside the pre-built ROCm container (no docker wrapper).
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/train_sae_FP16.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/trained_models}"

B="\033[1m"; C="\033[36m"; G="\033[32m"; Y="\033[33m"; R="\033[0m"
hr()  { echo -e "${C}────────────────────────────────────────────────────${R}"; }
ask() { local v=$1 p=$2 d=$3; echo -en "${B}${p}${R} [${Y}${d}${R}]: "; read -r i; printf -v "$v" '%s' "${i:-$d}"; }

# ── sanity check ─────────────────────────────────────────────────────────────
if [[ ! -f "${SCRIPT}" ]]; then
  echo -e "Training script not found at ${SCRIPT}"; exit 1
fi

hr; echo -e "${B}  SAE Training  —  ROCm + FP16${R}"; hr

# ── 1. Model ─────────────────────────────────────────────────────────────────
# Hyperparameters sourced from published configs:
#   LlamaScope (arxiv 2410.20526): TopK k=50, lr=8e-4, warm=10K, batch=4096, ctx=1024
#   GemmaScope  (arxiv 2408.05147): JumpReLU→TopK, lr=7e-5, warm=1K,  batch=4096, ctx=1024
echo -e "\n${B}Model:${R}"
echo "  1) meta-llama/Llama-3.1-8B   + FineWeb-Edu  (LlamaScope config)"
echo "  2) meta-llama/Llama-3.1-70B  + FineWeb-Edu  (LlamaScope config, multi-GPU recommended)"
echo "  3) Qwen/Qwen3-8B             + FineWeb-Edu  (LlamaScope config)"
echo "  4) google/gemma-3-4b         + C4           (GemmaScope config)"
ask CHOICE "Choice" "1"

case "${CHOICE}" in
  # LlamaScope (arxiv 2410.20526) — 8x expansion, TopK k=50, lr=8e-4
  1) MODEL="meta-llama/Llama-3.1-8B";  DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=4096;  N_LAYERS=32; D_SAE=32768;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50;  LR_RELU=5e-5; LR_WARM_RELU=0; L1=5.0; BATCH=4096; CTX=1024; DTYPE=bfloat16 ;;
  # LlamaScope (arxiv 2410.20526) — 8x expansion, TopK k=50, lr=8e-4
  2) MODEL="meta-llama/Llama-3.1-70B"; DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=8192;  N_LAYERS=80; D_SAE=65536;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50;  LR_RELU=5e-5; LR_WARM_RELU=0; L1=5.0; BATCH=4096; CTX=1024; DTYPE=bfloat16 ;;
  # LlamaScope-style (same architecture as 8B; no published Qwen3 SAE suite yet)
  3) MODEL="Qwen/Qwen3-8B";            DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"; D_IN=4096;  N_LAYERS=36; D_SAE=32768;  LR_TOPK=8e-4; LR_WARM_TOPK=10000; K=50;  LR_RELU=5e-5; LR_WARM_RELU=0; L1=5.0; BATCH=4096; CTX=1024; DTYPE=bfloat16 ;;
  # GemmaScope (arxiv 2408.05147) — 16K features, lr=7e-5, cosine warmup 1K steps
  4) MODEL="google/gemma-3-4b";        DATASET="allenai/c4";                DS_CFG="en";           D_IN=2560;  N_LAYERS=34; D_SAE=16384;  LR_TOPK=7e-5; LR_WARM_TOPK=1000;  K=50;  LR_RELU=7e-5; LR_WARM_RELU=1000; L1=5.0; BATCH=4096; CTX=1024; DTYPE=bfloat16 ;;
  *) echo "Invalid."; exit 1 ;;
esac
MODEL_SHORT="${MODEL##*/}"

# ── 2. Architecture ───────────────────────────────────────────────────────────
echo -e "\n${B}Architecture:${R}"
echo "  1) topk  — hard k-sparse (LlamaScope/GemmaScope default; no L1 tuning)"
echo "  2) relu  — L1 penalty    (standard SAE; requires tuning l1_coefficient)"
ask ARCH_CHOICE "Architecture" "1"

case "${ARCH_CHOICE}" in
  1)
    ARCH=topk
    LR="${LR_TOPK}"; LR_WARM="${LR_WARM_TOPK}"
    ARCH_EXTRA="--k ${K}"
    ARCH_SUMMARY="topk  k=${K}"
    ;;
  2)
    ARCH=relu
    LR="${LR_RELU}"; LR_WARM="${LR_WARM_RELU}"
    ask L1 "L1 coefficient" "${L1}"
    ARCH_EXTRA="--l1-coeff ${L1}"
    ARCH_SUMMARY="relu  l1=${L1}"
    ;;
  *) echo "Invalid."; exit 1 ;;
esac

# ── 4. Layer(s) ───────────────────────────────────────────────────────────────
MID=$(( N_LAYERS / 2 ))
echo -e "\n${B}Layer(s):${R}"
echo "  Enter a single layer index (0–$(( N_LAYERS - 1 ))), or 'all' to train every layer"
ask LAYER_INPUT "Layer" "${MID}"

if [[ "${LAYER_INPUT}" == "all" ]]; then
  LAYERS=( $(seq 0 $(( N_LAYERS - 1 ))) )
else
  LAYERS=( "${LAYER_INPUT}" )
fi

# ── 5. GPU ────────────────────────────────────────────────────────────────────
echo -e "\n${B}GPU:${R}"
echo "  Enter a single GPU index (0–7), or 'all' to spread layers across all 8 GPUs"
ask GPU_INPUT "GPU" "0"

# ── 6. Training tokens ────────────────────────────────────────────────────────
echo -e "\n${B}Training tokens:${R}"
echo "  1) 200M   — quick test  (~2-4h per layer on 1 GPU)"
echo "  2) 500M   — standard   (~5-10h per layer)  [LlamaScope-scale]"
echo "  3)   1B   — full run   (~10-20h per layer)"
ask TOK_CHOICE "Tokens" "1"
case "${TOK_CHOICE}" in
  1) TOKENS=200000000  ;;
  2) TOKENS=500000000  ;;
  3) TOKENS=1000000000 ;;
  *) TOKENS=200000000  ;;
esac

# ── 7. W&B ───────────────────────────────────────────────────────────────────
ask WANDB_PROJECT "W&B project" "efficient_sae"
ask NO_WANDB      "Disable W&B? (y/n)" "n"
WANDB_FLAG=""; [[ "${NO_WANDB,,}" == "y" ]] && WANDB_FLAG="--no-wandb"

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_STEPS=$(( TOKENS / BATCH ))
LR_DECAY=$(( TOTAL_STEPS / 5 ))
hr
printf "  %-18s %s\n" "Model:"       "${MODEL}"
printf "  %-18s %s\n" "Dataset:"     "${DATASET} [${DS_CFG}]"
printf "  %-18s %s\n" "Arch:"        "${ARCH_SUMMARY}  d_sae=${D_SAE} (d_in=${D_IN} × $(( D_SAE / D_IN ))x)"
printf "  %-18s %s\n" "LR:"          "${LR}  warmup=${LR_WARM}  decay=${LR_DECAY} (last 20%)"
printf "  %-18s %s\n" "Batch/ctx:"   "${BATCH} tokens / ${CTX} ctx"
printf "  %-18s %s\n" "Tokens:"      "${TOKENS}  (${TOTAL_STEPS} steps)"
printf "  %-18s %s\n" "Layers:"      "${LAYERS[*]}"
printf "  %-18s %s\n" "GPU(s):"      "${GPU_INPUT}"
printf "  %-18s %s\n" "dtype:"       "${DTYPE}"
printf "  %-18s %s\n" "W&B:"         "${WANDB_PROJECT}"
hr

ask CONFIRM "Launch? (y/n)" "y"
[[ "${CONFIRM,,}" != "y" ]] && echo "Aborted." && exit 0

# ── Run directory ─────────────────────────────────────────────────────────────
# One folder per run, auto-numbered, with everything needed to recover it:
#   <output-dir>/<model>/run<N>/
#     config.json        run-level hyperparameters (tokens, dtype, lr, arch, ...)
#     logs/layer<L>.txt  per-layer training logs
#     L0/ L1/ ... LN/    per-layer SAE checkpoints
RUN_ROOT="${OUTPUT_DIR}/${MODEL_SHORT}"
mkdir -p "${RUN_ROOT}"
RUN_NUM=1
while [[ -e "${RUN_ROOT}/run${RUN_NUM}" ]]; do RUN_NUM=$(( RUN_NUM + 1 )); done
RUN_DIR="${RUN_ROOT}/run${RUN_NUM}"
mkdir -p "${RUN_DIR}/logs"

# Group every layer of this run together in W&B (Python uses the same scheme).
export WANDB_RUN_GROUP="${MODEL_SHORT}/run${RUN_NUM}"

# Architecture-specific sparsity field for the config file.
if [[ "${ARCH}" == "topk" ]]; then
  SPARSITY_JSON="\"k\": ${K}"
else
  SPARSITY_JSON="\"l1_coeff\": ${L1}"
fi

# Layers as a JSON array, e.g. [0, 1, 2].
LAYERS_JSON=$(printf '%s, ' "${LAYERS[@]}"); LAYERS_JSON="[${LAYERS_JSON%, }]"

cat > "${RUN_DIR}/config.json" <<EOF
{
  "model": "${MODEL}",
  "model_short": "${MODEL_SHORT}",
  "dataset": "${DATASET}",
  "dataset_config": "${DS_CFG}",
  "arch": "${ARCH}",
  ${SPARSITY_JSON},
  "d_in": ${D_IN},
  "d_sae": ${D_SAE},
  "expansion": $(( D_SAE / D_IN )),
  "training_tokens": ${TOKENS},
  "training_steps": ${TOTAL_STEPS},
  "lr": "${LR}",
  "lr_warm_up_steps": ${LR_WARM},
  "lr_decay_steps": ${LR_DECAY},
  "lr_scheduler": "constant",
  "batch_size": ${BATCH},
  "context_size": ${CTX},
  "dtype": "${DTYPE}",
  "layers": ${LAYERS_JSON},
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${WANDB_RUN_GROUP}",
  "created": "$(date +%Y%m%d_%H%M%S)"
}
EOF

echo -e "\n${G}Run dir:${R}  ${RUN_DIR}"
echo -e "${G}Config:${R}   ${RUN_DIR}/config.json"

# ── Launch ────────────────────────────────────────────────────────────────────
N_GPUS=8

run_layer() {
  local layer=$1 gpu=$2
  local hook="blocks.${layer}.hook_resid_post"

  "${PYTHON}" "${SCRIPT}" \
    --model            "${MODEL}" \
    --dataset          "${DATASET}" \
    --dataset-config   "${DS_CFG}" \
    --hook-name        "${hook}" \
    --d-in             "${D_IN}" \
    --d-sae            "${D_SAE}" \
    --arch             "${ARCH}" \
    ${ARCH_EXTRA} \
    --lr               "${LR}" \
    --lr-warm-up-steps "${LR_WARM}" \
    --lr-decay-steps   "${LR_DECAY}" \
    --lr-scheduler     "constant" \
    --batch-size       "${BATCH}" \
    --context-size     "${CTX}" \
    --training-tokens  "${TOKENS}" \
    --dtype            "${DTYPE}" \
    --device           "cuda:${gpu}" \
    --llm-device       "cuda:${gpu}" \
    --run-dir          "${RUN_DIR}" \
    --wandb-project    "${WANDB_PROJECT}" \
    ${WANDB_FLAG}
}

if [[ "${#LAYERS[@]}" -eq 1 ]]; then
  # Single layer — stream to terminal and tee into the run's log folder.
  GPU="${GPU_INPUT:-0}"
  LOG="${RUN_DIR}/logs/layer${LAYERS[0]}.txt"
  echo -e "\n${G}Training layer ${LAYERS[0]} on cuda:${GPU}  (log: ${LOG})${R}"
  run_layer "${LAYERS[0]}" "${GPU}" 2>&1 | tee "${LOG}"

else
  # All layers — spawn in background, assign GPUs round-robin
  echo -e "\n${G}Launching ${#LAYERS[@]} layers across GPUs...${R}"
  for i in "${!LAYERS[@]}"; do
    layer="${LAYERS[$i]}"
    if [[ "${GPU_INPUT}" == "all" ]]; then
      gpu=$(( i % N_GPUS ))
    else
      gpu="${GPU_INPUT}"
    fi
    LOG="${RUN_DIR}/logs/layer${layer}.txt"
    echo -e "  Layer ${layer}  →  cuda:${gpu}  (log: ${LOG})"
    run_layer "${layer}" "${gpu}" > "${LOG}" 2>&1 &
  done

  echo -e "\n${G}All ${#LAYERS[@]} jobs launched.${R}"
  echo "Monitor with:  tail -f ${RUN_DIR}/logs/layer<N>.txt"
  echo "Or:            rocm-smi --showuse"
  wait
  echo -e "\n${G}All layers done.${R}"
fi
