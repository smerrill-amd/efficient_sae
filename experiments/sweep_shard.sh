#!/usr/bin/env bash
# sweep_shard.sh — split the k-sweep + lr×batch sweeps across multiple GPUs/servers.
#
# Enumerates EVERY training job once (k-sweep per precision + every lr×batch cell per
# precision), then runs only the jobs assigned to THIS shard
# (job_index % NSHARDS == SHARD). Each job delegates to the existing sweep_k.sh /
# sweep_lr_batch.sh (so train+eval, idempotency, run-tagging, offline/local-data all
# behave identically) but is scoped to a single precision (k-sweep) or a single
# (precision, lr, batch) cell (lr×batch), and pinned to ONE physical GPU.
#
# WHY shard by job, not by hand: lr/batch are runner-global so each cell is its own
# process; k-sweep is one shared-model run per precision. Round-robin over all of them
# balances the load and — because outputs live on the shared /wekafs filesystem with
# unique per-cell run dirs and idempotent skipping — you can run the SAME command on
# every GPU/server with only SHARD/GPU_PHYS changed and nothing collides or recomputes.
#
# Usage (run one per free GPU; SHARD must be unique, NSHARDS = total GPUs):
#   # this server, GPU 1:
#   tmux new -d -s sweep0 "cd $PWD && SHARD=0 NSHARDS=3 GPU_PHYS=1 bash sweep_shard.sh"
#   # other server, GPU 0 and GPU 1:
#   tmux new -d -s sweep1 "cd $PWD && SHARD=1 NSHARDS=3 GPU_PHYS=0 bash sweep_shard.sh"
#   tmux new -d -s sweep2 "cd $PWD && SHARD=2 NSHARDS=3 GPU_PHYS=1 bash sweep_shard.sh"
#
# Knobs (env): MODEL, PRECISIONS, TRAINING_TOKENS, LRS, BATCHES, WIDTH, K (lr×batch),
#   KSWEEP_WIDTH, KS (k-sweep), N_CHECKPOINTS, CHECKPOINTS, EVALS, DO_KSWEEP, DO_LRBS,
#   LOCAL_DATA, HF_OFFLINE, DRY_RUN.
#   DRY_RUN=1 SHARD=0 NSHARDS=3 bash sweep_shard.sh   # print this shard's job list only
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SHARD="${SHARD:?set SHARD (0-based index of this GPU)}"
NSHARDS="${NSHARDS:?set NSHARDS (total number of GPUs across all servers)}"
GPU_PHYS="${GPU_PHYS:-0}"        # physical GPU index to pin on THIS box

# ── What to sweep ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-pythia}"
PRECISIONS="${PRECISIONS:-fp16 fp8te}"
TRAINING_TOKENS="${TRAINING_TOKENS:-100000000}"
DO_KSWEEP="${DO_KSWEEP:-1}"
DO_LRBS="${DO_LRBS:-1}"

# lr×batch grid (must match sweep_lr_batch.sh defaults so dirs line up).
LRS="${LRS:-1e-4 3e-4 1e-3 3e-3 1e-2}"
BATCHES="${BATCHES:-2048 4096 8192 16384}"
WIDTH="${WIDTH:-65536}"
K="${K:-80}"

# k-sweep grid.
KSWEEP_WIDTH="${KSWEEP_WIDTH:-65536}"
KS="${KS:-50,100,250,500,1000,2500}"

# Lighter defaults than the standalone scripts: frontier points only (no intermediate
# checkpoints, eval the final). Override if you want the full evolution.
N_CHECKPOINTS="${N_CHECKPOINTS:-0}"
CHECKPOINTS="${CHECKPOINTS:-final}"
EVALS="${EVALS:-core,sparse_probing}"

# Disconnect-proofing: default to the local Pile shard + offline HF for training (the
# FP8-TE path especially benefits from no mid-run HF streaming). Eval still goes online.
LOCAL_DATA="${LOCAL_DATA:-/wekafs/smerrill/data/pile-uncopyrighted/train/00.jsonl.zst}"
HF_OFFLINE="${HF_OFFLINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ -e "${LOCAL_DATA}" ]] || { echo "[shard] WARNING: LOCAL_DATA '${LOCAL_DATA}' not found; set LOCAL_DATA= to stream from HF."; }

# ── Build the global, deterministic job list ──────────────────────────────────
declare -a JOBS=()
if [[ "${DO_KSWEEP}" == "1" ]]; then
  for prec in ${PRECISIONS}; do JOBS+=( "ksweep:${prec}" ); done
fi
if [[ "${DO_LRBS}" == "1" ]]; then
  for prec in ${PRECISIONS}; do
    for lr in ${LRS}; do
      for bs in ${BATCHES}; do
        JOBS+=( "lrbs:${prec}:${lr}:${bs}" )
      done
    done
  done
fi

TOTAL="${#JOBS[@]}"
echo "============================================================"
echo "  sweep_shard  SHARD=${SHARD}/${NSHARDS}  phys GPU=${GPU_PHYS}  model=${MODEL}"
echo "  precisions=[${PRECISIONS}]  tokens=${TRAINING_TOKENS}"
echo "  total jobs=${TOTAL}  (k-sweep=${DO_KSWEEP}, lr×batch=${DO_LRBS})"
echo "  this shard runs jobs: $(for i in $(seq 0 $((TOTAL-1))); do [[ $((i % NSHARDS)) -eq ${SHARD} ]] && echo -n "$i "; done)"
echo "============================================================"

# Common env for both delegated scripts: pin a single physical GPU (TE needs cuda:0
# in-process) and pass the shared knobs.
common_env=(
  GPU=0
  CUDA_VISIBLE_DEVICES="${GPU_PHYS}"
  HIP_VISIBLE_DEVICES="${GPU_PHYS}"
  TRAINING_TOKENS="${TRAINING_TOKENS}"
  N_CHECKPOINTS="${N_CHECKPOINTS}"
  CHECKPOINTS="${CHECKPOINTS}"
  EVALS="${EVALS}"
  LOCAL_DATA="${LOCAL_DATA}"
)
if [[ "${HF_OFFLINE}" == "1" ]]; then
  common_env+=( HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 )
fi

run_job() {  # $1 = job spec
  local spec="$1"
  IFS=':' read -r kind a b c <<< "${spec}"
  case "${kind}" in
    ksweep)
      local prec="${a}"
      echo ""
      echo ">>> [shard ${SHARD}] KSWEEP  prec=${prec}  model=${MODEL}"
      [[ "${DRY_RUN}" == "1" ]] && { echo "    (dry-run)"; return 0; }
      env "${common_env[@]}" \
          MODELS="${MODEL}" PRECISIONS="${prec}" \
          WIDTH="${KSWEEP_WIDTH}" KS="${KS}" \
          bash "${SCRIPT_DIR}/sweep_k.sh"
      ;;
    lrbs)
      local prec="${a}" lr="${b}" bs="${c}"
      echo ""
      echo ">>> [shard ${SHARD}] LRBS  prec=${prec} lr=${lr} bs=${bs}  model=${MODEL}"
      [[ "${DRY_RUN}" == "1" ]] && { echo "    (dry-run)"; return 0; }
      env "${common_env[@]}" \
          MODEL="${MODEL}" PRECISIONS="${prec}" \
          WIDTH="${WIDTH}" K="${K}" LRS="${lr}" BATCHES="${bs}" \
          bash "${SCRIPT_DIR}/sweep_lr_batch.sh"
      ;;
    *) echo "[shard] unknown job kind '${kind}' in '${spec}'" >&2; return 1 ;;
  esac
}

rc=0
for i in "${!JOBS[@]}"; do
  if [[ $((i % NSHARDS)) -eq ${SHARD} ]]; then
    run_job "${JOBS[$i]}" || { rc=1; echo "[shard ${SHARD}] job ${i} (${JOBS[$i]}) FAILED — continuing."; }
  fi
done

echo ""
echo "============================================================"
echo "[shard ${SHARD}] $(date)  done (rc=${rc}). Analyze with"
echo "  notebooks/ksweep_frontier.ipynb  and  notebooks/lr_batch_analysis.ipynb"
echo "============================================================"
exit "${rc}"
