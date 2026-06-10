#!/usr/bin/env bash
# sweep_k.sh — FP8-vs-FP16-vs-authors k-sweep, for BOTH SAEBench models.
#
# For each model (gemma-2-2b, pythia-160m-deduped) and each precision (FP16, FP8),
# train one shared-model BatchTopK run over the whole k grid at a single width, with a
# handful of checkpoints, then evaluate every (k, checkpoint) with the SAEBench evals
# that need no credentials (core + sparse_probing). The companion notebook
# notebooks/ksweep_frontier.ipynb then plots our FP8 / FP16 points against the authors'
# published BatchTopK frontier (adamkarvonen/sae_bench_results_0125) for the same model.
#
# This box has ONE GPU, so the four train runs (2 models x 2 precisions) run
# SEQUENTIALLY on ${GPU}. Each run trains all k at once (one model, one forward pass).
#
# Phases (PHASE=train|eval|all, default all):
#   train  -> 4 training runs (gemma/pythia x fp16/fp8)
#   eval   -> evals every member x checkpoint of whatever has been trained
#
# Usage / overrides (env):
#   ./sweep_k.sh                                  # full thing, defaults below
#   PHASE=train ./sweep_k.sh                      # just train
#   PHASE=eval CHECKPOINTS=final ./sweep_k.sh     # only final ckpt -> fast frontier
#   MODELS="gemma" PRECISIONS="fp8" ./sweep_k.sh  # a slice
#   KS="50,100,250" WIDTH=65536 ./sweep_k.sh
#   DRY_RUN=1 ./sweep_k.sh                         # print the plan, run nothing
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TRAIN="${SCRIPT_DIR}/train_saebench_replication.py"
EVAL_SH="${SCRIPT_DIR}/eval_saebench.sh"

# Load secrets (.env) — gemma-2-2b is gated; W&B key too.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Knobs ─────────────────────────────────────────────────────────────────────
MODELS="${MODELS:-gemma pythia}"
PRECISIONS="${PRECISIONS:-fp16 fp8}"
WIDTH="${WIDTH:-65536}"                       # 2^16 — has an authors release for both models
KS="${KS:-50,100,250,500,1000,2500}"
N_CHECKPOINTS="${N_CHECKPOINTS:-4}"           # 4 intermediate + final = 5 total checkpoints
GPU="${GPU:-0}"
PHASE="${PHASE:-all}"
EVALS="${EVALS:-core,sparse_probing}"
CHECKPOINTS="${CHECKPOINTS:-all}"             # 'all' (5/ckpt, evolution) or 'final' (fast frontier)
SAE_DTYPE="${SAE_DTYPE:-float32}"
DTYPE="${DTYPE:-bfloat16}"
TRAINING_TOKENS="${TRAINING_TOKENS:-}"        # empty -> SAEBench 500M
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
DRY_RUN="${DRY_RUN:-0}"

TS="$(date +%Y%m%d_%H%M%S)"

# fp16/fp8 -> the run-dir suffix train_saebench_replication.py produces.
suffix_for() { [[ "$1" == "fp8" ]] && echo "_fp8" || echo ""; }

echo "============================================================"
echo "  k-sweep  models=[${MODELS}]  precisions=[${PRECISIONS}]"
echo "  width=${WIDTH}  ks=[${KS}]  checkpoints=${N_CHECKPOINTS} (+final)"
echo "  phase=${PHASE}  evals=[${EVALS}]  eval_ckpts=${CHECKPOINTS}  gpu=${GPU}"
echo "  output=${OUTPUT_DIR}"
echo "============================================================"

# ── Train phase ───────────────────────────────────────────────────────────────
train_phase() {
  for model in ${MODELS}; do
    for prec in ${PRECISIONS}; do
      local fp8_flag=(); [[ "${prec}" == "fp8" ]] && fp8_flag=( --fp8 )
      local sfx; sfx="$(suffix_for "${prec}")"
      local log="${OUTPUT_DIR}/sweepk_train_${model}_${prec}_${TS}.log"
      local args=(
        --model         "${model}"
        --gpu           "${GPU}"
        --output-dir    "${OUTPUT_DIR}"
        --widths        "${WIDTH}"
        --ks            "${KS}"
        --n-checkpoints "${N_CHECKPOINTS}"
        --dtype         "${DTYPE}"
        --sae-dtype     "${SAE_DTYPE}"
        "${fp8_flag[@]}"
      )
      [[ -n "${TRAINING_TOKENS}" ]] && args+=( --training-tokens "${TRAINING_TOKENS}" )
      [[ "${DRY_RUN}" == "1" ]]     && args+=( --dry-run )
      echo ""
      echo ">>> TRAIN ${model}/${prec}  -> results/saebench_${model}${sfx}  (log: ${log})"
      if [[ "${DRY_RUN}" == "1" ]]; then
        "${PYTHON}" "${TRAIN}" "${args[@]}"
      else
        "${PYTHON}" "${TRAIN}" "${args[@]}" 2>&1 | tee "${log}"
      fi
    done
  done
}

# ── Eval phase ────────────────────────────────────────────────────────────────
eval_phase() {
  IFS=',' read -ra KARR <<< "${KS}"
  for model in ${MODELS}; do
    for prec in ${PRECISIONS}; do
      local sfx; sfx="$(suffix_for "${prec}")"
      local run_dir="${OUTPUT_DIR}/saebench_${model}${sfx}"
      if [[ ! -d "${run_dir}" ]]; then
        echo ">>> SKIP eval ${model}/${prec}: ${run_dir} not found (train it first)"
        continue
      fi
      for k in "${KARR[@]}"; do
        local member="w${WIDTH}_k${k}"
        if [[ ! -d "${run_dir}/${member}" ]]; then
          echo ">>> SKIP ${model}/${prec}/${member}: not trained"
          continue
        fi
        echo ""
        echo ">>> EVAL ${model}/${prec}/${member}  (ckpts=${CHECKPOINTS})"
        if [[ "${DRY_RUN}" == "1" ]]; then
          RUN_DIR="${run_dir}" MEMBER="${member}" CHECKPOINTS="${CHECKPOINTS}" \
            EVALS="${EVALS}" GPU="${GPU}" DRY_RUN=1 "${EVAL_SH}"
        else
          RUN_DIR="${run_dir}" MEMBER="${member}" CHECKPOINTS="${CHECKPOINTS}" \
            EVALS="${EVALS}" GPU="${GPU}" "${EVAL_SH}"
        fi
      done
    done
  done
}

case "${PHASE}" in
  train) train_phase ;;
  eval)  eval_phase ;;
  all)   train_phase; eval_phase ;;
  *) echo "unknown PHASE='${PHASE}' (use train|eval|all)"; exit 2 ;;
esac

echo ""
echo "============================================================"
echo "  k-sweep ${PHASE} done. Analyze with notebooks/ksweep_frontier.ipynb"
echo "============================================================"
