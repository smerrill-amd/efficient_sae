#!/usr/bin/env bash
# sweep_lr_batch.sh — learning-rate x batch-size sweep for BatchTopK SAEs.
#
# Trains one BatchTopK SAE per (lr, batch) cell of a 2-D grid, at a single
# width/k, then evaluates each with the no-credentials SAEBench evals
# (core + sparse_probing). The companion notebook
# notebooks/lr_batch_analysis.ipynb renders heatmaps of every quality metric
# over the (lr x batch) plane, per precision.
#
# WHY this is its own driver (vs. sweep_k.sh):
#   lr and batch-size are *runner-global* — they can't be shared across a single
#   multi-SAE run the way (width, k) can. So each grid cell is a separate train
#   run. To keep that affordable, this sweep fixes ONE (width, k) and saves a
#   handful of checkpoints (N_CHECKPOINTS=4 intermediate + final; set 0 for
#   final-only). The eval phase defaults to the FINAL checkpoint.
#
# BATCH SIZE IS ARCHITECTURAL IN BatchTopK:
#   top-k is taken over the whole batch, so changing --batch-size changes the
#   selection statistics, not just the optimizer. Two ways to study it:
#     GROUP_SIZE=""      (default) raw — batch affects BOTH optimization and the
#                        BatchTopK pool. Shows the real, coupled effect.
#     GROUP_SIZE=2048    ghost-batch control — top-k is taken within fixed 2048-
#                        sample groups regardless of --batch-size, so the selection
#                        pool is held constant and batch becomes a *pure optimization*
#                        knob. Must divide every value in BATCHES.
#   Run it twice (once each) to separate the two effects cleanly.
#
# The token-consistent schedule in train_saebench_replication.py already rescales
# LR warmup/decay by --batch-size, so every cell sees the same 500M-token schedule.
#
# Phases (PHASE=train|eval|all, default all):
#   train -> one run per (precision, lr, batch)
#   eval  -> evaluate the FINAL checkpoint of each trained cell
#
# Usage / overrides (env):
#   ./sweep_lr_batch.sh                                   # full grid, defaults below
#   MODEL=gemma PRECISIONS=fp8 ./sweep_lr_batch.sh
#   LRS="1e-4 3e-4 1e-3" BATCHES="2048 4096 8192 16384" ./sweep_lr_batch.sh
#   GROUP_SIZE=2048 RUN_GROUP=ghost ./sweep_lr_batch.sh   # ghost-batch control study
#   TRAINING_TOKENS=100000000 ./sweep_lr_batch.sh         # quick 100M-token scan
#   PHASE=eval ./sweep_lr_batch.sh                        # just (re)eval finals
#   GPU=1 BATCHES="8192 16384" ./sweep_lr_batch.sh        # a slice on another GPU
#   DRY_RUN=1 ./sweep_lr_batch.sh                         # print the plan, run nothing
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
MODEL="${MODEL:-gemma}"                        # one model (lr/batch dynamics study)
PRECISIONS="${PRECISIONS:-fp8}"                # "fp8", "fp16", or "fp16 fp8"
WIDTH="${WIDTH:-65536}"
K="${K:-80}"                                   # single representative sparsity
LRS="${LRS:-1e-4 3e-4 1e-3 3e-3 1e-2}"
BATCHES="${BATCHES:-2048 4096 8192 16384}"
GROUP_SIZE="${GROUP_SIZE:-}"                   # "" raw; e.g. 2048 = ghost-batch control
RUN_GROUP="${RUN_GROUP:-}"                     # extra run-tag prefix (e.g. "ghost"), keeps studies apart
N_CHECKPOINTS="${N_CHECKPOINTS:-4}"            # 4 intermediate + final = 5 checkpoints (set 0 for final-only)
GPU="${GPU:-0}"
SEED="${SEED:-0}"                             # RNG seed for every train run (reproducible by default)
PHASE="${PHASE:-all}"
EVALS="${EVALS:-core,sparse_probing}"
CHECKPOINTS="${CHECKPOINTS:-final}"            # final is the point for an lr/batch grid
SAE_DTYPE="${SAE_DTYPE:-float32}"
DTYPE="${DTYPE:-bfloat16}"
TRAINING_TOKENS="${TRAINING_TOKENS:-}"         # empty -> SAEBench 500M
LOCAL_DATA="${LOCAL_DATA:-}"                    # local .jsonl(.zst) path -> no HF streaming (disconnect-proof)
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
DRY_RUN="${DRY_RUN:-0}"

TS="$(date +%Y%m%d_%H%M%S)"

# precision -> the run-dir suffix train_saebench_replication.py produces.
suffix_for() {
  case "$1" in
    fp8te) echo "_fp8te" ;;   # TransformerEngine fp8 path (--fp8-te)
    fp8)   echo "_fp8" ;;     # torch._scaled_mm/emulated fp8 path (--fp8)
    *)     echo "" ;;          # fp16/bf16 baseline
  esac
}

# run-tag for a grid cell: [<group>_]lr<lr>_bs<bs>[_g<group_size>]
# Encodes everything the notebook needs to parse back out of the dir name.
tag_for() {
  local lr="$1" bs="$2"
  local t="lr${lr}_bs${bs}"
  [[ -n "${GROUP_SIZE}" ]] && t="${t}_g${GROUP_SIZE}"
  [[ -n "${RUN_GROUP}"  ]] && t="${RUN_GROUP}_${t}"
  echo "${t}"
}

echo "============================================================"
echo "  lr x batch sweep   model=${MODEL}  precisions=[${PRECISIONS}]"
echo "  width=${WIDTH}  k=${K}   lrs=[${LRS}]   batches=[${BATCHES}]"
echo "  group_size=${GROUP_SIZE:-<none/raw>}  run_group=${RUN_GROUP:-<none>}"
echo "  checkpoints=${N_CHECKPOINTS} (+final)  tokens=${TRAINING_TOKENS:-500M}"
echo "  phase=${PHASE}  evals=[${EVALS}]  eval_ckpts=${CHECKPOINTS}  gpu=${GPU}  seed=${SEED}"
echo "  output=${OUTPUT_DIR}"
echo "============================================================"

# ── Train phase ───────────────────────────────────────────────────────────────
train_phase() {
  for prec in ${PRECISIONS}; do
    local fp8_flag=()
    case "${prec}" in
      fp8te)
        fp8_flag=( --fp8-te )
        [[ -n "${FP8_SCALING:-}" ]] && fp8_flag+=( --fp8-scaling "${FP8_SCALING}" )
        [[ -n "${FP8_RECIPE:-}" ]]  && fp8_flag+=( --fp8-recipe "${FP8_RECIPE}" )
        ;;
      fp8) fp8_flag=( --fp8 ) ;;
    esac
    local sfx; sfx="$(suffix_for "${prec}")"
    for lr in ${LRS}; do
      for bs in ${BATCHES}; do
        local tag; tag="$(tag_for "${lr}" "${bs}")"
        local log="${OUTPUT_DIR}/sweeplrbs_train_${MODEL}_${prec}_${tag}_${TS}.log"
        local args=(
          --model         "${MODEL}"
          --gpu           "${GPU}"
          --output-dir    "${OUTPUT_DIR}"
          --widths        "${WIDTH}"
          --ks            "${K}"
          --lr            "${lr}"
          --batch-size    "${bs}"
          --run-tag       "${tag}"
          --n-checkpoints "${N_CHECKPOINTS}"
          --dtype         "${DTYPE}"
          --sae-dtype     "${SAE_DTYPE}"
          --seed          "${SEED}"
          "${fp8_flag[@]}"
        )
        [[ -n "${GROUP_SIZE}" ]]      && args+=( --topk-group-size "${GROUP_SIZE}" )
        [[ -n "${TRAINING_TOKENS}" ]] && args+=( --training-tokens "${TRAINING_TOKENS}" )
        [[ -n "${LOCAL_DATA}" ]]      && args+=( --local-data "${LOCAL_DATA}" )
        [[ "${DRY_RUN}" == "1" ]]     && args+=( --dry-run )
        echo ""
        echo ">>> TRAIN ${prec} lr=${lr} bs=${bs}  -> results/saebench_${MODEL}${sfx}_${tag}  (log: ${log})"
        if [[ "${DRY_RUN}" == "1" ]]; then
          "${PYTHON}" "${TRAIN}" "${args[@]}"
        else
          "${PYTHON}" "${TRAIN}" "${args[@]}" 2>&1 | tee "${log}"
        fi
      done
    done
  done
}

# ── Eval phase ────────────────────────────────────────────────────────────────
eval_phase() {
  local member="w${WIDTH}_k${K}"
  for prec in ${PRECISIONS}; do
    local sfx; sfx="$(suffix_for "${prec}")"
    for lr in ${LRS}; do
      for bs in ${BATCHES}; do
        local tag; tag="$(tag_for "${lr}" "${bs}")"
        local run_dir="${OUTPUT_DIR}/saebench_${MODEL}${sfx}_${tag}"
        if [[ ! -d "${run_dir}/${member}" ]]; then
          echo ">>> SKIP eval ${prec} lr=${lr} bs=${bs}: ${run_dir}/${member} not trained"
          continue
        fi
        echo ""
        echo ">>> EVAL ${prec} lr=${lr} bs=${bs}  ${run_dir}/${member}  (ckpts=${CHECKPOINTS})"
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
echo "  lr x batch ${PHASE} done. Analyze with notebooks/lr_batch_analysis.ipynb"
echo "============================================================"
