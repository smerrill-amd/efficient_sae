#!/usr/bin/env bash
# estimate_relu_train_time.sh — how long does a ReLU SAE take, by model size?
#
# Trains ONE single ReLU/Standard (L1) SAE for a SHORT token budget on a grid of
# Llama / Qwen / Gemma models of increasing size (largest = 8B, never bigger),
# with the built-in wall-clock profiler ON. For every model it records:
#   * the hidden dim used to train the SAE (d_in at the hook = the SAE input dim)
#   * the split of compute time between the LLM forward pass and SAE training
#   * the measured per-batch throughput
# so the companion notebook can extrapolate to the full 500M-token production
# run in shell_scripts/train_interactive.sh.
#
# Nothing is saved: --n-checkpoints 0 --no-save-final (this is a timing probe).
#
# Models (d_in / n_layers from the HF configs; hook layer = round(HOOK_FRAC*n)):
#   qwen0.6b  Qwen/Qwen3-0.6B        d_in=1024  n=28
#   qwen1.7b  Qwen/Qwen3-1.7B        d_in=2048  n=28
#   qwen4b    Qwen/Qwen3-4B          d_in=2560  n=36
#   qwen8b    Qwen/Qwen3-8B          d_in=4096  n=36   *in train_interactive.sh*
#   llama1b   meta-llama/Llama-3.2-1B d_in=2048 n=16
#   llama3b   meta-llama/Llama-3.2-3B d_in=3072 n=28
#   llama8b   meta-llama/Llama-3.1-8B d_in=4096 n=32   *in train_interactive.sh*
#   gemma1b   google/gemma-3-1b-pt   d_in=1152  n=26
#   gemma4b   google/gemma-3-4b-pt   d_in=2560  n=34   *in train_interactive.sh*
#
# Override anything via env, e.g.:
#   BENCH_TOKENS=1000000 GPU=1 MODELS="qwen0.6b llama1b gemma1b" ./estimate_relu_train_time.sh
#   DICT_MULT=8 ./estimate_relu_train_time.sh
set -uo pipefail   # NOTE: no -e — one failing / unsupported model must not abort the grid.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/train_sae_FP16.py}"

# Load secrets (.env) — Llama / Gemma are gated and need an HF token.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${ENV_FILE}"; set +a
fi

# Reduce allocator fragmentation (matches train_interactive.sh / estimate_train_time.sh).
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# ── Benchmark settings (env-overridable) ─────────────────────────────────────
# BENCH_TOKENS is the SHORT probe budget. Default 100k tokens (~49 steps @ batch
# 2048): a fast probe so the whole grid finishes well inside a ~1.5h window. The
# profiler isolates the one-time model load and reports per-batch forward/SAE
# times, so even this short run extrapolates fine. Bump it (e.g. 1000000) for a
# steadier measurement if you have more time. The full production run is
# PROD_TOKENS (see below).
# NOTE: this is a *token* budget, not a step count. 100k STEPS would be
# 100000*batch = ~205M tokens/model — far too long for a 1.5h grid; hence 100k
# tokens. To cap by step count instead, pass --profile-steps via the runner.
BENCH_TOKENS="${BENCH_TOKENS:-100000}"
BATCH="${BATCH:-2048}"          # train_interactive.sh defaults
CTX="${CTX:-1024}"
DTYPE="${DTYPE:-bfloat16}"       # activation / buffer dtype
# ReLU/Standard (L1) SAE: use fp32 master weights (the robust recipe). Native
# bf16 weights crash this arch's backward ("Found dtype Float but expected
# BFloat16") because the L1/MSE loss mixes fp32 with bf16 params and there is no
# autocast/GradScaler in that path. With fp32 weights, autocast still runs the
# matmuls in bf16, so the measured timing stays representative.
SAE_DTYPE="${SAE_DTYPE:-float32}"
DICT_MULT="${DICT_MULT:-16}"     # d_sae = DICT_MULT * d_in (16x mirrors gemma/qwen prod)
HOOK_FRAC_PCT="${HOOK_FRAC_PCT:-67}"  # hook layer = round(0.67 * n_layers), like the prod rule
GPU="${GPU:-0}"

# Full production run (shell_scripts/train_interactive.sh) — used by the notebook
# to extrapolate; recorded into the CSV so the projection is self-contained.
PROD_TOKENS="${PROD_TOKENS:-500000000}"

# Default grid spans ~0.6B → 8B across Qwen + Gemma. Llama-3.2 models are gated
# and not authorized for this HF token (403), so they're excluded by default;
# add them back via MODELS=... if you have access. Trim/extend via MODELS=...
MODELS="${MODELS:-qwen0.6b qwen1.7b qwen4b qwen8b gemma1b gemma4b}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/relu_scale_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"
CSV="${RESULTS_DIR}/results.csv"
echo "model_key,model,family,params_b,d_in,n_layers,hook_layer,hook,arch,bench_d_sae,dict_mult,dataset,dataset_config,bench_tokens,batch,ctx,prod_tokens,in_train_interactive,prod_d_sae,prod_arch,status,wall_seconds,timing_json" > "${CSV}"

# ── Per-model descriptors ────────────────────────────────────────────────────
# Fields set: MODEL D_IN N_LAYERS DATASET DS_CFG FAMILY PARAMS_B
#             IN_PROD(0/1) PROD_DSAE PROD_ARCH  (prod_* only meaningful when IN_PROD=1)
set_model_params() {
  IN_PROD=0; PROD_DSAE=0; PROD_ARCH="-"
  case "$1" in
    qwen0.6b) MODEL="Qwen/Qwen3-0.6B";        D_IN=1024; N_LAYERS=28; FAMILY=qwen;  PARAMS_B=0.6
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    qwen1.7b) MODEL="Qwen/Qwen3-1.7B";        D_IN=2048; N_LAYERS=28; FAMILY=qwen;  PARAMS_B=1.7
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    qwen4b)   MODEL="Qwen/Qwen3-4B";          D_IN=2560; N_LAYERS=36; FAMILY=qwen;  PARAMS_B=4
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    qwen8b)   MODEL="Qwen/Qwen3-8B";          D_IN=4096; N_LAYERS=36; FAMILY=qwen;  PARAMS_B=8
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"
              IN_PROD=1; PROD_DSAE=65536;  PROD_ARCH=batchtopk ;;
    llama1b)  MODEL="meta-llama/Llama-3.2-1B"; D_IN=2048; N_LAYERS=16; FAMILY=llama; PARAMS_B=1
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    llama3b)  MODEL="meta-llama/Llama-3.2-3B"; D_IN=3072; N_LAYERS=28; FAMILY=llama; PARAMS_B=3
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT" ;;
    llama8b)  MODEL="meta-llama/Llama-3.1-8B"; D_IN=4096; N_LAYERS=32; FAMILY=llama; PARAMS_B=8
              DATASET="HuggingFaceFW/fineweb-edu"; DS_CFG="sample-10BT"
              IN_PROD=1; PROD_DSAE=131072; PROD_ARCH=batchtopk ;;
    gemma1b)  MODEL="google/gemma-3-1b-pt";   D_IN=1152; N_LAYERS=26; FAMILY=gemma; PARAMS_B=1
              DATASET="allenai/c4"; DS_CFG="en" ;;
    gemma4b)  MODEL="google/gemma-3-4b-pt";   D_IN=2560; N_LAYERS=34; FAMILY=gemma; PARAMS_B=4
              DATASET="allenai/c4"; DS_CFG="en"
              IN_PROD=1; PROD_DSAE=$(( 16 * 2560 )); PROD_ARCH=batchtopk ;;
    *) echo "Unknown model key '$1'"; return 1 ;;
  esac
}

echo "============================================================"
echo "  ReLU SAE train-time benchmark (by model size)"
echo "  bench_tokens=${BENCH_TOKENS}  batch=${BATCH}  ctx=${CTX}  dtype=${DTYPE}"
echo "  dict_mult=${DICT_MULT}x  hook=round(${HOOK_FRAC_PCT}%*n_layers)  gpu=cuda:${GPU}"
echo "  models=[${MODELS}]"
echo "  project full run -> ${PROD_TOKENS} tokens"
echo "  results -> ${RESULTS_DIR}"
echo "============================================================"

for m in ${MODELS}; do
  set_model_params "${m}" || continue

  # hook layer = round(HOOK_FRAC_PCT% * n_layers), clamped to [0, n_layers-1]
  HOOK_LAYER=$(( (HOOK_FRAC_PCT * N_LAYERS + 50) / 100 ))
  (( HOOK_LAYER >= N_LAYERS )) && HOOK_LAYER=$(( N_LAYERS - 1 ))
  HOOK="blocks.${HOOK_LAYER}.hook_resid_post"
  D_SAE=$(( DICT_MULT * D_IN ))

  RUN_DIR="${RESULTS_DIR}/${m}"
  mkdir -p "${RUN_DIR}/logs"
  LOG="${RUN_DIR}/logs/train.txt"
  TIMING_JSON="${RUN_DIR}/L${HOOK_LAYER}/timing_profile.json"

  ARGS=(
    --model            "${MODEL}"
    --dataset          "${DATASET}"
    --dataset-config   "${DS_CFG}"
    --d-in             "${D_IN}"
    --d-sae            "${D_SAE}"
    --arch             relu
    --batch-size       "${BATCH}"
    --context-size     "${CTX}"
    --training-tokens  "${BENCH_TOKENS}"
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

  echo
  echo "────────────────────────────────────────────────────────────"
  echo "  ${m}  (${MODEL})  d_in=${D_IN}  ${HOOK}  d_sae=${D_SAE} (${DICT_MULT}x)"
  echo "────────────────────────────────────────────────────────────"
  START=${SECONDS}
  "${PYTHON}" "${SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG}"
  STATUS=${PIPESTATUS[0]}
  WALL=$(( SECONDS - START ))

  echo "${m},${MODEL},${FAMILY},${PARAMS_B},${D_IN},${N_LAYERS},${HOOK_LAYER},${HOOK},relu,${D_SAE},${DICT_MULT},${DATASET},${DS_CFG},${BENCH_TOKENS},${BATCH},${CTX},${PROD_TOKENS},${IN_PROD},${PROD_DSAE},${PROD_ARCH},${STATUS},${WALL},${TIMING_JSON}" >> "${CSV}"
  if [[ "${STATUS}" -eq 0 ]]; then
    echo "  ✓ ${m} done in ${WALL}s"
  else
    echo "  ✗ ${m} FAILED (exit ${STATUS}) after ${WALL}s — see ${LOG}"
  fi
done

echo
echo "============================================================"
echo "  Benchmark complete."
echo "  Summary CSV : ${CSV}"
echo "  Plot it     : open notebooks/relu_scale_train_times.ipynb and set"
echo "                RESULTS_DIR = \"${RESULTS_DIR}\""
echo "============================================================"
column -t -s, "${CSV}" 2>/dev/null || cat "${CSV}"
