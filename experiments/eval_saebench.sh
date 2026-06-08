#!/usr/bin/env bash
# eval_saebench.sh — run the SAEBench eval suite on one of OUR trained SAEs and
# dump JSON results for comparison against the published baselines.
#
# Defaults reproduce the Neuronpedia config you sent
# (gemma-2-2b / blocks.12.hook_resid_post, 65k width, batch_topk, trainer_2):
#   MEMBER=w65536_k80  (trainer_2 -> achieved L0 ~84)
#
# The companion notebook notebooks/saebench_compare.ipynb then compares these
# results to the authors' (adamkarvonen/sae_bench_results_0125 / Neuronpedia).
#
# Usage / overrides (env):
#   ./eval_saebench.sh                                   # final SAE, default evals
#   MEMBER=w16384_k40 ./eval_saebench.sh                 # a different config
#   CHECKPOINTS=all ./eval_saebench.sh                   # every training checkpoint
#   CHECKPOINTS=2441,24414,final ./eval_saebench.sh      # specific steps
#   EVALS=core,sparse_probing ./eval_saebench.sh         # subset of evals
#   EVALS=all ./eval_saebench.sh                         # everything supported
#   GPU=1 ./eval_saebench.sh
#   DRY_RUN=1 ./eval_saebench.sh                         # print plan, no eval
#
# NOTE: CHECKPOINTS=all needs a run trained WITH intermediate checkpoints, e.g.
#   N_CHECKPOINTS=10 WIDTHS=65536 KS=80 ./train_saebench.sh
# (the default training run uses N_CHECKPOINTS=0 -> only the final checkpoint).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/eval_saebench.py}"

# Load secrets (.env) — gemma-2-2b is gated and needs an HF token.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${ENV_FILE}"; set +a
fi

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Knobs (env-overridable) ──────────────────────────────────────────────────
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/results/saebench_gemma}"
MEMBER="${MEMBER:-w65536_k80}"
CHECKPOINTS="${CHECKPOINTS:-final}"
EVALS="${EVALS:-core,sparse_probing,absorption,scr,tpp}"
GPU="${GPU:-0}"
TAG="${TAG:-mysae}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/saebench_eval/${MEMBER}}"

args=(
  --run-dir     "${RUN_DIR}"
  --member      "${MEMBER}"
  --checkpoints "${CHECKPOINTS}"
  --evals       "${EVALS}"
  --gpu         "${GPU}"
  --tag         "${TAG}"
  --output-dir  "${OUTPUT_DIR}"
)
[[ -n "${LLM_BATCH_SIZE:-}" ]] && args+=( --llm-batch-size "${LLM_BATCH_SIZE}" )
[[ -n "${LLM_DTYPE:-}" ]]      && args+=( --llm-dtype "${LLM_DTYPE}" )
[[ "${FORCE_RERUN:-0}" == "1" ]]      && args+=( --force-rerun )
[[ "${SAVE_ACTIVATIONS:-0}" == "1" ]] && args+=( --save-activations )
[[ "${DRY_RUN:-0}" == "1" ]]          && args+=( --dry-run )

mkdir -p "${OUTPUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTPUT_DIR}/eval_saebench_${TS}.log"

echo "============================================================"
echo "  SAEBench eval   member=${MEMBER}  checkpoints=${CHECKPOINTS}"
echo "  evals=[${EVALS}]  gpu=${GPU}"
echo "  run=${RUN_DIR}"
echo "  output=${OUTPUT_DIR}"
echo "  log=${LOG}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT}" "${args[@]}"
  exit $?
fi

"${PYTHON}" "${SCRIPT}" "${args[@]}" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
