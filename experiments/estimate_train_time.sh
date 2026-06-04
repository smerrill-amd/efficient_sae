#!/usr/bin/env bash
# estimate_train_time.sh — quick per-architecture SAE training-time benchmark.
#
# Trains ONE single SAE (--sweep none) for a short token budget (default 1M) on
# each MODEL x ARCH combination, then records how long each took. Use it to
# estimate how long a full run will take and to compare architectures.
#
#   Models : Qwen3-8B, gemma-3-4b-pt   (same defaults as shell_scripts/train_interactive.sh)
#   Archs  : topk, batchtopk, standard (L1), jumprelu
#
# Each run writes a timing_profile.json (the built-in profiler, on by default)
# under its run dir, plus a row in results.csv. Point the companion notebook
# (notebooks/compare_train_times.ipynb) at the results dir to plot everything.
#
# Override anything via env, e.g.:
#   TOKENS=2000000 GPU=1 MODELS="qwen" ARCHS="topk jumprelu" ./estimate_train_time.sh
set -uo pipefail   # NOTE: no -e — one failing combo must not abort the matrix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/train_sae_FP16.py}"

# Load secrets (.env) — gemma-3-4b-pt is gated and needs an HF token.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${ENV_FILE}"; set +a
fi

# Reduce allocator fragmentation (matches train_interactive.sh).
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# ── Benchmark settings (env-overridable) ─────────────────────────────────────
TOKENS="${TOKENS:-1000000}"     # short run just to measure throughput
BATCH="${BATCH:-2048}"          # train_interactive.sh defaults
CTX="${CTX:-1024}"
DTYPE="${DTYPE:-bfloat16}"      # activation / buffer dtype
SAE_DTYPE="${SAE_DTYPE:-bfloat16}"  # SAE weight dtype (bf16 = native bf16 training, no GradScaler)
K="${K:-32}"                    # TopK / BatchTopK L0 target
GPU="${GPU:-0}"
MODELS="${MODELS:-qwen gemma}"
ARCHS="${ARCHS:-topk batchtopk standard jumprelu}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"
CSV="${RESULTS_DIR}/results.csv"
echo "model,model_name,arch,hook,d_in,d_sae,tokens,batch,ctx,status,wall_seconds,timing_json" > "${CSV}"

# ── Per-model defaults (copied from shell_scripts/train_interactive.sh) ──────
set_model_params() {
  case "$1" in
    qwen)
      MODEL="Qwen/Qwen3-8B";        D_IN=4096; HOOK_LAYER=24; D_SAE=65536;
      LR=3e-4; DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    gemma)
      MODEL="google/gemma-3-4b-pt"; D_IN=2560; HOOK_LAYER=23; D_SAE=$(( 16 * 2560 ));
      LR=3e-4; DATASET="allenai/c4";                DS_CFG="en" ;;
    *) echo "Unknown model key '$1' (expected qwen|gemma)"; return 1 ;;
  esac
}

echo "============================================================"
echo "  SAE train-time benchmark"
echo "  tokens=${TOKENS}  batch=${BATCH}  ctx=${CTX}  dtype=${DTYPE}  gpu=cuda:${GPU}"
echo "  models=[${MODELS}]  archs=[${ARCHS}]"
echo "  results -> ${RESULTS_DIR}"
echo "============================================================"

for m in ${MODELS}; do
  set_model_params "${m}" || continue
  HOOK="blocks.${HOOK_LAYER}.hook_resid_post"
  for arch in ${ARCHS}; do
    RUN_DIR="${RESULTS_DIR}/${m}/${arch}"
    mkdir -p "${RUN_DIR}/logs"
    LOG="${RUN_DIR}/logs/train.txt"
    TIMING_JSON="${RUN_DIR}/L${HOOK_LAYER}/timing_profile.json"

    ARGS=(
      --model            "${MODEL}"
      --dataset          "${DATASET}"
      --dataset-config   "${DS_CFG}"
      --d-in             "${D_IN}"
      --d-sae            "${D_SAE}"
      --arch             "${arch}"
      --lr               "${LR}"
      --batch-size       "${BATCH}"
      --context-size     "${CTX}"
      --training-tokens  "${TOKENS}"
      --dtype            "${DTYPE}"
      --sae-dtype        "${SAE_DTYPE}"
      --device           "cuda:${GPU}"
      --llm-device       "cuda:${GPU}"
      --sweep            none
      --hook-name        "${HOOK}"
      --run-dir          "${RUN_DIR}"
      --n-checkpoints    0
      --no-save-final
      --no-wandb
    )
    # k only applies to the hard-sparsity architectures.
    case "${arch}" in
      topk|batchtopk) ARGS+=( --k "${K}" ) ;;
    esac

    echo
    echo "────────────────────────────────────────────────────────────"
    echo "  ${m} / ${arch}  @ ${HOOK}  (d_sae=${D_SAE})"
    echo "────────────────────────────────────────────────────────────"
    START=${SECONDS}
    "${PYTHON}" "${SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG}"
    STATUS=${PIPESTATUS[0]}
    WALL=$(( SECONDS - START ))

    echo "${m},${MODEL},${arch},${HOOK},${D_IN},${D_SAE},${TOKENS},${BATCH},${CTX},${STATUS},${WALL},${TIMING_JSON}" >> "${CSV}"
    if [[ "${STATUS}" -eq 0 ]]; then
      echo "  ✓ ${m}/${arch} done in ${WALL}s"
    else
      echo "  ✗ ${m}/${arch} FAILED (exit ${STATUS}) after ${WALL}s — see ${LOG}"
    fi
  done
done

echo
echo "============================================================"
echo "  Benchmark complete."
echo "  Summary CSV : ${CSV}"
echo "  Plot it     : open notebooks/compare_train_times.ipynb and set"
echo "                RESULTS_DIR = \"${RESULTS_DIR}\""
echo "============================================================"
column -t -s, "${CSV}" 2>/dev/null || cat "${CSV}"
