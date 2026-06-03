#!/usr/bin/env bash
# train_interactive.sh — launch SAE training from the initial sweep.
#
# Pick the MODEL, the ARCH (batchtopk/topk), and a MODE:
#   1) single   one SAE at one layer/k                       (SAETrainingRunner)
#   2) layers   one SAE per layer, all sharing one model     (MultiSAETrainingRunner)
#   3) k / L0   several k values at one layer, shared model  (MultiSAETrainingRunner)
# The sweep modes load the model ONCE and run a single forward pass per batch,
# multiplexing activations to every SAE — far cheaper than one process per layer.
#
#   Model           hook                       d_sae        k options        tokens  batch  lr     ctx
#   Qwen3-8B        blocks.24.hook_resid_post  65_536       32/64/128/256    500M    2048   3e-4   1024
#   Llama-3.1-8B    blocks.20.hook_resid_post  131_072      32/64/128/256    500M    2048   2e-4   1024
#   gemma-3-4b      blocks.23.hook_resid_post  16*d_in      32/64/128/256    500M    2048   3e-4   1024
#
# Layer rule: Qwen3-8B uses blocks.24 (36 layers); gemma uses round(0.67*n_layers)
# = round(0.67*34) = 23; Llama is fixed at blocks.20.
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

hr; echo -e "${B}  SAE Sweep  —  TopK / BatchTopK on resid_post${R}"; hr

# ── 1. Model ─────────────────────────────────────────────────────────────────
echo -e "\n${B}Model:${R}"
echo "  1) Qwen3-8B       blocks.24  d_sae=65536   lr=3e-4   (FineWeb-Edu)"
echo "  2) Llama-3.1-8B   blocks.20  d_sae=131072  lr=2e-4   (FineWeb-Edu)"
echo "  3) gemma-3-4b     blocks.23  d_sae=16*d_in lr=3e-4   (C4)"
ask MODEL_CHOICE "Model" "1"

case "${MODEL_CHOICE}" in
  1) MODEL="Qwen/Qwen3-8B";          D_IN=4096; N_LAYERS=36; HOOK_LAYER=24; D_SAE=65536;          LR=3e-4; DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
  2) MODEL="meta-llama/Llama-3.1-8B"; D_IN=4096; N_LAYERS=32; HOOK_LAYER=20; D_SAE=131072;         LR=2e-4; DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
  3) MODEL="google/gemma-3-4b";      D_IN=2560; N_LAYERS=34; HOOK_LAYER=23; D_SAE=$(( 16 * 2560 )); LR=3e-4; DATASET="allenai/c4";                DS_CFG="en" ;;
  *) echo "Invalid."; exit 1 ;;
esac
MODEL_SHORT="${MODEL##*/}"

# ── 2. Architecture ──────────────────────────────────────────────────────────
echo -e "\n${B}Architecture:${R}"
echo "  1) batchtopk   (per-batch hard sparsity)"
echo "  2) topk        (per-token hard sparsity)"
ask ARCH_CHOICE "Arch" "1"
case "${ARCH_CHOICE}" in
  1) ARCH=batchtopk ;;
  2) ARCH=topk ;;
  *) echo "Invalid."; exit 1 ;;
esac

# ── 3. Mode ──────────────────────────────────────────────────────────────────
echo -e "\n${B}Mode:${R}"
echo "  1) single   one SAE at one layer/k"
echo "  2) layers   one SAE per layer (shared model, one forward pass)"
echo "  3) k / L0   several k values at one layer (shared model)"
ask MODE_CHOICE "Mode" "1"

# pick a single TopK k from the standard set
pick_k() {
  echo -e "\n${B}TopK k (L0 target):${R}"
  echo "  1) 32"; echo "  2) 64"; echo "  3) 128"; echo "  4) 256"
  local c; ask c "k" "1"
  case "${c}" in
    1) K=32 ;; 2) K=64 ;; 3) K=128 ;; 4) K=256 ;;
    *) echo "Invalid."; exit 1 ;;
  esac
}

SWEEP="none"; K=""; K_VALUES=""; LAYERS=""; LAYER=""
case "${MODE_CHOICE}" in
  1)  # single
    SWEEP="none"
    pick_k
    echo -e "\n${B}Layer (resid_post):${R}  index 0–$(( N_LAYERS - 1 ))  (default ${HOOK_LAYER})"
    ask LAYER "Layer" "${HOOK_LAYER}"
    if ! [[ "${LAYER}" =~ ^[0-9]+$ ]] || (( LAYER < 0 || LAYER >= N_LAYERS )); then
      echo "Invalid layer '${LAYER}'."; exit 1
    fi
    ;;
  2)  # layers sweep
    SWEEP="layers"
    pick_k
    echo -e "\n${B}Layers:${R}  comma list (e.g. '0,8,16,24') or 'all' (0–$(( N_LAYERS - 1 )))"
    ask LAYERS "Layers" "all"
    ;;
  3)  # k sweep
    SWEEP="k"
    echo -e "\n${B}Layer (resid_post):${R}  index 0–$(( N_LAYERS - 1 ))  (default ${HOOK_LAYER})"
    ask LAYER "Layer" "${HOOK_LAYER}"
    if ! [[ "${LAYER}" =~ ^[0-9]+$ ]] || (( LAYER < 0 || LAYER >= N_LAYERS )); then
      echo "Invalid layer '${LAYER}'."; exit 1
    fi
    echo -e "\n${B}k values:${R}  comma list of L0 targets"
    ask K_VALUES "k values" "32,64,128,256"
    ;;
  *) echo "Invalid."; exit 1 ;;
esac

# ── 4. GPU ───────────────────────────────────────────────────────────────────
echo -e "\n${B}GPU:${R}  single index 0–7"
ask GPU "GPU" "0"

# ── Fixed sweep settings ─────────────────────────────────────────────────────
TOKENS=500000000
BATCH=2048
CTX=1024
DTYPE=bfloat16
LR_WARM=0
TOTAL_STEPS=$(( TOKENS / BATCH ))
LR_DECAY=$(( TOTAL_STEPS / 5 ))

# W&B project is model-specific so each model's runs live in their own project.
WANDB_PROJECT="efficient_sae-${MODEL_SHORT}"
export WANDB_PROJECT
WANDB_FLAG=""; [[ "${NO_WANDB:-0}" == "1" ]] && WANDB_FLAG="--no-wandb"

# ── Summary ───────────────────────────────────────────────────────────────────
hr
printf "  %-18s %s\n" "Model:"     "${MODEL}"
printf "  %-18s %s\n" "Dataset:"   "${DATASET} [${DS_CFG}]"
printf "  %-18s %s\n" "Arch:"      "${ARCH}  d_sae=${D_SAE} (d_in=${D_IN} × $(( D_SAE / D_IN ))x)"
printf "  %-18s %s\n" "Mode:"      "${SWEEP}"
case "${SWEEP}" in
  none)   printf "  %-18s %s\n" "Layer/k:" "blocks.${LAYER}.hook_resid_post  k=${K}" ;;
  layers) printf "  %-18s %s\n" "Layers:"  "${LAYERS}  (resid_post, k=${K})" ;;
  k)      printf "  %-18s %s\n" "Layer/k:" "blocks.${LAYER}.hook_resid_post  k=[${K_VALUES}]" ;;
esac
printf "  %-18s %s\n" "LR:"        "${LR}  warmup=${LR_WARM}  decay=${LR_DECAY} (last 20%)"
printf "  %-18s %s\n" "Batch/ctx:" "${BATCH} tokens / ${CTX} ctx"
printf "  %-18s %s\n" "Tokens:"    "${TOKENS}  (${TOTAL_STEPS} steps)"
printf "  %-18s %s\n" "GPU:"       "cuda:${GPU}"
printf "  %-18s %s\n" "dtype:"     "${DTYPE}"
printf "  %-18s %s\n" "W&B:"       "${WANDB_PROJECT}"
hr
if [[ "${SWEEP}" != "none" ]]; then
  echo -e "${Y}Note:${R} sweep modes train every SAE in ONE process; ensure all SAEs"
  echo -e "      fit in cuda:${GPU} memory (model + N×SAE params/optimizer states)."
fi

ask CONFIRM "Launch? (y/n)" "y"
[[ "${CONFIRM,,}" != "y" ]] && echo "Aborted." && exit 0

# ── Run directory ─────────────────────────────────────────────────────────────
# One folder per run, auto-numbered, with everything needed to recover it:
#   <output-dir>/<model>/run<N>/
#     config.json                 run-level hyperparameters
#     logs/<...>.txt              training log(s)
#     L<layer>/ | checkpoints/    SAE checkpoints (layout depends on mode)
RUN_ROOT="${OUTPUT_DIR}/${MODEL_SHORT}"
mkdir -p "${RUN_ROOT}"
RUN_NUM=1
while [[ -e "${RUN_ROOT}/run${RUN_NUM}" ]]; do RUN_NUM=$(( RUN_NUM + 1 )); done
RUN_DIR="${RUN_ROOT}/run${RUN_NUM}"
mkdir -p "${RUN_DIR}/logs"

# Group this run together in W&B.
export WANDB_RUN_GROUP="${MODEL_SHORT}/run${RUN_NUM}"

# Record the mode-specific selection as JSON.
case "${SWEEP}" in
  none)   SEL="\"k\": ${K}, \"layer\": ${LAYER}" ;;
  layers) SEL="\"k\": ${K}, \"layers\": \"${LAYERS}\"" ;;
  k)      SEL="\"layer\": ${LAYER}, \"k_values\": \"${K_VALUES}\"" ;;
esac

cat > "${RUN_DIR}/config.json" <<EOF
{
  "model": "${MODEL}",
  "model_short": "${MODEL_SHORT}",
  "dataset": "${DATASET}",
  "dataset_config": "${DS_CFG}",
  "arch": "${ARCH}",
  "sweep": "${SWEEP}",
  ${SEL},
  "hook_template": "blocks.{layer}.hook_resid_post",
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
  "gpu": "${GPU}",
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${WANDB_RUN_GROUP}",
  "created": "$(date +%Y%m%d_%H%M%S)"
}
EOF

echo -e "\n${G}Run dir:${R}  ${RUN_DIR}"
echo -e "${G}Config:${R}   ${RUN_DIR}/config.json"

# ── Common args (shared by every mode) ────────────────────────────────────────
COMMON=(
  --model            "${MODEL}"
  --dataset          "${DATASET}"
  --dataset-config   "${DS_CFG}"
  --d-in             "${D_IN}"
  --d-sae            "${D_SAE}"
  --arch             "${ARCH}"
  --lr               "${LR}"
  --lr-warm-up-steps "${LR_WARM}"
  --lr-decay-steps   "${LR_DECAY}"
  --lr-scheduler     "constant"
  --batch-size       "${BATCH}"
  --context-size     "${CTX}"
  --training-tokens  "${TOKENS}"
  --dtype            "${DTYPE}"
  --device           "cuda:${GPU}"
  --llm-device       "cuda:${GPU}"
  --run-dir          "${RUN_DIR}"
  --wandb-project    "${WANDB_PROJECT}"
)
[[ -n "${WANDB_FLAG}" ]] && COMMON+=("${WANDB_FLAG}")

# ── Launch ────────────────────────────────────────────────────────────────────
case "${SWEEP}" in
  none)
    HOOK="blocks.${LAYER}.hook_resid_post"
    LOG="${RUN_DIR}/logs/layer${LAYER}.txt"
    echo -e "\n${G}Training ${MODEL_SHORT} ${HOOK} (k=${K}) on cuda:${GPU}  (log: ${LOG})${R}\n"
    "${PYTHON}" "${SCRIPT}" "${COMMON[@]}" \
      --sweep none --hook-name "${HOOK}" --k "${K}" 2>&1 | tee "${LOG}"
    ;;
  layers)
    LOG="${RUN_DIR}/logs/sweep_layers.txt"
    echo -e "\n${G}Layer sweep [${LAYERS}] (k=${K}) on cuda:${GPU}  (log: ${LOG})${R}\n"
    "${PYTHON}" "${SCRIPT}" "${COMMON[@]}" \
      --sweep layers --layers "${LAYERS}" --n-layers "${N_LAYERS}" --k "${K}" 2>&1 | tee "${LOG}"
    ;;
  k)
    HOOK="blocks.${LAYER}.hook_resid_post"
    LOG="${RUN_DIR}/logs/sweep_k_L${LAYER}.txt"
    echo -e "\n${G}k sweep [${K_VALUES}] @ ${HOOK} on cuda:${GPU}  (log: ${LOG})${R}\n"
    "${PYTHON}" "${SCRIPT}" "${COMMON[@]}" \
      --sweep k --hook-name "${HOOK}" --k-values "${K_VALUES}" 2>&1 | tee "${LOG}"
    ;;
esac

echo -e "\n${G}Done.${R}  Checkpoints + logs in ${RUN_DIR}"
