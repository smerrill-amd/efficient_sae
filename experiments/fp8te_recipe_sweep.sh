#!/usr/bin/env bash
# fp8te_recipe_sweep.sh — FP8-TransformerEngine recipe x scaling study on gemma-2-2b.
#
# Trains the SAME SAE config (default gemma-2-2b, w65536_k80, 100M tokens) once for
# every combination of TE fp8 RECIPE x SCALING, then runs the credential-free SAEBench
# suite (core + sparse_probing) on each. The companion notebook
# notebooks/fp8te_recipe_analysis.ipynb tabulates how each metric moves across the grid.
#
#   recipes  : hybrid (E4M3 fwd / E5M2 bwd) | e4m3 (symmetric) | e5m2 (symmetric)
#   scalings : delayed (rolling amax history) | current (per-tensor dynamic)
#   -> 3 x 2 = 6 FP8-TE runs.
#
# Each run lands in results/saebench_<model>_fp8te_<recipe>_<scaling>/ (the _fp8te
# suffix is added by the trainer; <recipe>_<scaling> is our --run-tag), so every cell
# is separate and parseable by the notebook.
#
# Env knobs (all overridable):
#   MODEL=gemma  WIDTH=65536  K=80  TRAINING_TOKENS=100000000
#   RECIPES="hybrid e4m3 e5m2"  SCALINGS="delayed current"
#   GPU_PHYS=0  EVALS=core,sparse_probing  N_CHECKPOINTS=0  CHECKPOINTS=final  SEED=0
#   LOCAL_DATA=<pile shard>  HF_OFFLINE=1  SKIP_TRAIN=0  SKIP_EVAL=0  DRY_RUN=0
#
# Examples:
#   GPU_PHYS=1 ./fp8te_recipe_sweep.sh                       # full 6-cell study on phys GPU 1
#   SCALINGS=delayed ./fp8te_recipe_sweep.sh                 # delayed-only (3 cells)
#   DRY_RUN=1 ./fp8te_recipe_sweep.sh                        # print the plan, do nothing
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TRAIN="${SCRIPT_DIR}/train_saebench_replication.py"
EVAL_SH="${SCRIPT_DIR}/eval_saebench.sh"
OUT="${OUT:-${SCRIPT_DIR}/results}"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Knobs ─────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-gemma}"
WIDTH="${WIDTH:-65536}"
K="${K:-80}"
MEMBER="w${WIDTH}_k${K}"
TRAINING_TOKENS="${TRAINING_TOKENS:-100000000}"
RECIPES="${RECIPES:-hybrid e4m3 e5m2}"
SCALINGS="${SCALINGS:-delayed current}"
GPU_PHYS="${GPU_PHYS:-0}"          # physical GPU to pin (TE needs a single visible GPU)
EVALS="${EVALS:-core,sparse_probing}"
N_CHECKPOINTS="${N_CHECKPOINTS:-0}"
CHECKPOINTS="${CHECKPOINTS:-final}"
SEED="${SEED:-0}"
LOCAL_DATA="${LOCAL_DATA:-/wekafs/smerrill/data/pile-uncopyrighted/train/00.jsonl.zst}"
HF_OFFLINE="${HF_OFFLINE:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"
[[ -e "${LOCAL_DATA}" ]] || echo "[recipe] WARNING: LOCAL_DATA '${LOCAL_DATA}' not found (training would hit HF)."

# TransformerEngine runs its fp8 GEMM on the process's *current* CUDA device, so pin
# ONE physical GPU (becomes cuda:0 in-process) and pass --gpu 0.
export CUDA_VISIBLE_DEVICES="${GPU_PHYS}" HIP_VISIBLE_DEVICES="${GPU_PHYS}"

echo "============================================================"
echo "  FP8-TE recipe x scaling study"
echo "  model=${MODEL}  member=${MEMBER}  tokens=${TRAINING_TOKENS}  seed=${SEED}"
echo "  recipes=[${RECIPES}]  scalings=[${SCALINGS}]  -> $(echo ${RECIPES} | wc -w)x$(echo ${SCALINGS} | wc -w) cells"
echo "  phys GPU=${GPU_PHYS}  evals=[${EVALS}]  out=${OUT}"
echo "============================================================"

train_one() {  # $1=recipe  $2=scaling
  local recipe="$1" scaling="$2" tag dir log
  tag="${recipe}_${scaling}"
  dir="${OUT}/saebench_${MODEL}_fp8te_${tag}"
  if [[ -f "${dir}/${MEMBER}/cfg.json" ]]; then
    echo "[recipe] TRAIN skip ${tag} (final SAE already at ${dir}/${MEMBER})"; return 0
  fi
  [[ "${SKIP_TRAIN}" == "1" ]] && { echo "[recipe] TRAIN skip ${tag} (SKIP_TRAIN=1)"; return 0; }
  log="${OUT}/recipe_train_${MODEL}_fp8te_${tag}_${TS}.log"
  echo "[recipe] TRAIN ${tag}  recipe=${recipe} scaling=${scaling}  -> ${dir}  (log ${log})"
  local args=(
    --model "${MODEL}" --gpu 0 --widths "${WIDTH}" --ks "${K}"
    --training-tokens "${TRAINING_TOKENS}" --fp8-te
    --fp8-recipe "${recipe}" --fp8-scaling "${scaling}"
    --run-tag "${tag}" --seed "${SEED}" --n-checkpoints "${N_CHECKPOINTS}"
    --no-wandb --output-dir "${OUT}"
  )
  [[ -e "${LOCAL_DATA}" ]] && args+=( --local-data "${LOCAL_DATA}" )
  if [[ "${DRY_RUN}" == "1" ]]; then echo "        ${PYTHON} ${TRAIN} ${args[*]}"; return 0; fi
  local off=()
  [[ "${HF_OFFLINE}" == "1" ]] && off=( HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 )
  env "${off[@]}" "${PYTHON}" "${TRAIN}" "${args[@]}" > "${log}" 2>&1 \
    || { echo "[recipe] TRAIN FAILED ${tag} (see ${log})"; return 1; }
}

eval_one() {  # $1=recipe  $2=scaling
  local recipe="$1" scaling="$2" tag dir
  tag="${recipe}_${scaling}"
  dir="${OUT}/saebench_${MODEL}_fp8te_${tag}"
  if [[ ! -f "${dir}/${MEMBER}/cfg.json" ]]; then echo "[recipe] EVAL skip ${tag} (no final SAE)"; return 0; fi
  [[ "${SKIP_EVAL}" == "1" ]] && { echo "[recipe] EVAL skip ${tag} (SKIP_EVAL=1)"; return 0; }
  echo "[recipe] EVAL ${tag}  [${EVALS}]"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "        RUN_DIR=${dir} MEMBER=${MEMBER} CHECKPOINTS=${CHECKPOINTS} EVALS=${EVALS} GPU=0 bash ${EVAL_SH}"; return 0
  fi
  RUN_DIR="${dir}" MEMBER="${MEMBER}" CHECKPOINTS="${CHECKPOINTS}" EVALS="${EVALS}" GPU=0 \
    bash "${EVAL_SH}" >> "${OUT}/recipe_eval_${MODEL}_fp8te_${TS}.log" 2>&1 \
    || echo "[recipe] EVAL FAILED ${tag} (see ${OUT}/recipe_eval_${MODEL}_fp8te_${TS}.log)"
}

# Train all cells first, then eval all (training is offline, eval needs online HF).
for recipe in ${RECIPES}; do
  for scaling in ${SCALINGS}; do
    train_one "${recipe}" "${scaling}"
  done
done
for recipe in ${RECIPES}; do
  for scaling in ${SCALINGS}; do
    eval_one "${recipe}" "${scaling}"
  done
done

echo "============================================================"
echo "[recipe] $(date)  done. Cells: saebench_${MODEL}_fp8te_<recipe>_<scaling>"
echo "  Analyse with notebooks/fp8te_recipe_analysis.ipynb"
echo "============================================================"
