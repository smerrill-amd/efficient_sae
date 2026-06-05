#!/usr/bin/env bash
# train_saebench.sh — replicate the SAEBench BatchTopK suite on TWO GPUs at once.
#
#   GPU 0 : gemma-2-2b           layer 12, 3 widths x 6 k = 18 SAEs (one shared run)
#   GPU 1 : pythia-160m-deduped  layer  8, 3 widths x 6 k = 18 SAEs (one shared run)
#
# Each model trains all its widths AND all its sparsities in a SINGLE process that
# loads the LLM once and multiplexes one forward pass to every SAE
# (MultiSAETrainingRunner). The two models run concurrently — one per GPU — so the
# whole SAEBench BatchTopK grid trains in one shot.
#
# Hyperparameters are pinned to the SAEBench release configs (see the header of
# train_saebench_replication.py): 500M tokens, lr 3e-4, 1k warmup, last-20% decay,
# The Pile, batch 2048, ctx 1024, auxk=1/32, BatchTopK threshold beta=0.999.
#
# Override anything via env, e.g.:
#   GEMMA_GPU=2 PYTHIA_GPU=3 ./train_saebench.sh
#   WIDTHS="4096,16384" SAE_DTYPE=bfloat16 ./train_saebench.sh   # if 65k x6 OOMs
#   MODELS="pythia" ./train_saebench.sh                          # just one model
#   DRY_RUN=1 ./train_saebench.sh                                # print configs, no train
set -uo pipefail   # no -e: one model failing must not kill the other.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/train_saebench_replication.py}"

# Load secrets (.env) — gemma-2-2b is gated and needs an HF token; W&B key too.
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${ENV_FILE}"; set +a
fi

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# ── Knobs (env-overridable) ──────────────────────────────────────────────────
GEMMA_GPU="${GEMMA_GPU:-0}"
PYTHIA_GPU="${PYTHIA_GPU:-1}"
MODELS="${MODELS:-gemma pythia}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
DTYPE="${DTYPE:-bfloat16}"
SAE_DTYPE="${SAE_DTYPE:-float32}"
WIDTHS="${WIDTHS:-}"            # empty -> script default 4096,16384,65536
KS="${KS:-}"                   # empty -> script default 20,40,80,160,320,640
N_CHECKPOINTS="${N_CHECKPOINTS:-0}"
NO_WANDB_FLAG=""; [[ "${NO_WANDB:-0}" == "1" ]] && NO_WANDB_FLAG="--no-wandb"
DRY_FLAG=""; [[ "${DRY_RUN:-0}" == "1" ]] && DRY_FLAG="--dry-run"

mkdir -p "${OUTPUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

LAST_PID=""
run_model() {  # $1=model key, $2=gpu ; backgrounds the run and sets LAST_PID.
  # NOTE: must run in the main shell (NOT via $(...)), otherwise the background
  # job would be a child of the command-substitution subshell and `wait` in the
  # parent could not see it. We return the pid through LAST_PID instead of stdout
  # so the status echo never pollutes the captured pid.
  local model="$1" gpu="$2"
  local log="${OUTPUT_DIR}/saebench_${model}_${TS}.log"
  local args=(
    --model         "${model}"
    --gpu           "${gpu}"
    --output-dir    "${OUTPUT_DIR}"
    --dataset       "monology/pile-uncopyrighted"
    --dtype         "${DTYPE}"
    --sae-dtype     "${SAE_DTYPE}"
    --n-checkpoints "${N_CHECKPOINTS}"
  )
  [[ -n "${WIDTHS}" ]] && args+=( --widths "${WIDTHS}" )
  [[ -n "${KS}" ]]     && args+=( --ks "${KS}" )
  [[ -n "${NO_WANDB_FLAG}" ]] && args+=( "${NO_WANDB_FLAG}" )
  [[ -n "${DRY_FLAG}" ]]      && args+=( "${DRY_FLAG}" )

  echo "  launching ${model} on cuda:${gpu}  (log: ${log})"
  "${PYTHON}" "${SCRIPT}" "${args[@]}" > "${log}" 2>&1 &
  LAST_PID=$!
}

echo "============================================================"
echo "  SAEBench BatchTopK replication (parallel, one model per GPU)"
echo "  models=[${MODELS}]  gemma->cuda:${GEMMA_GPU}  pythia->cuda:${PYTHIA_GPU}"
echo "  widths=[${WIDTHS:-default 4096,16384,65536}]  ks=[${KS:-default 20,40,80,160,320,640}]"
echo "  acts=${DTYPE}  sae=${SAE_DTYPE}  ckpts=${N_CHECKPOINTS}  output=${OUTPUT_DIR}"
echo "============================================================"

declare -a PIDS=() NAMES=()
for m in ${MODELS}; do
  case "${m}" in
    gemma)  run_model gemma  "${GEMMA_GPU}" ;;
    pythia) run_model pythia "${PYTHIA_GPU}" ;;
    *) echo "  skip unknown model '${m}'"; continue ;;
  esac
  PIDS+=("${LAST_PID}"); NAMES+=("${m}")
done

# Wait for every launched run and report each exit status.
RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ✓ ${NAMES[$i]} finished (pid ${PIDS[$i]})"
  else
    st=$?; RC=1
    echo "  ✗ ${NAMES[$i]} FAILED (pid ${PIDS[$i]}, exit ${st}) — see ${OUTPUT_DIR}/saebench_${NAMES[$i]}_${TS}.log"
  fi
done

echo "============================================================"
echo "  All launched runs complete. Outputs under ${OUTPUT_DIR}/saebench_<model>/"
echo "============================================================"
exit "${RC}"
