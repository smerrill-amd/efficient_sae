#!/usr/bin/env bash
# run_fp8te_lrbs.sh — disconnect-resilient FP8-TE lr×batch pipeline for one variant.
#
# Phase 1 TRAIN: fully offline (HF_HUB_OFFLINE + HF_DATASETS_OFFLINE) reading the Pile
#   from a LOCAL shard (--local-data), so a mid-run HF/network drop can't kill the long
#   (~12h) training. Model + tokenizer load from the HF cache.
# Phase 2 EVAL: online (queued right after train, same tmux session) — the SAEBench
#   sparse-probing `codeparrot/github-code` set is streaming-only, so eval needs live HF;
#   the other datasets + model come from cache. Eval is idempotent (re-runnable).
#
# Run each variant in its own tmux session, pinned to one GPU via HIP_VISIBLE_DEVICES
# (the TE path assumes cuda:0 in-process). Examples:
#   tmux new -d -s lrbs_fp8te_raw   "HIP_VISIBLE_DEVICES=0 GPU=0 bash run_fp8te_lrbs.sh"
#   tmux new -d -s lrbs_fp8te_ghost "HIP_VISIBLE_DEVICES=1 GPU=0 GROUP_SIZE=2048 RUN_GROUP=ghost bash run_fp8te_lrbs.sh"
set -uo pipefail

SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILE="${PILE:-/wekafs/smerrill/data/pile-uncopyrighted}"

# Common knobs (override via env). GROUP_SIZE / RUN_GROUP pass through for the ghost variant.
export MODEL="${MODEL:-gemma}" PRECISIONS="${PRECISIONS:-fp8te}"
export WIDTH="${WIDTH:-65536}" K="${K:-80}"
export LRS="${LRS:-1e-4 3e-4 1e-3}" BATCHES="${BATCHES:-2048 4096 8192 16384}"
export TRAINING_TOKENS="${TRAINING_TOKENS:-100000000}" N_CHECKPOINTS="${N_CHECKPOINTS:-0}"
export EVALS="${EVALS:-core,sparse_probing}" CHECKPOINTS="${CHECKPOINTS:-final}"
export GPU="${GPU:-0}"

echo "[fp8te-lrbs] $(date)  variant=${RUN_GROUP:-raw}  GPU=${GPU} HIP=${HIP_VISIBLE_DEVICES:-unset}"
echo "[fp8te-lrbs] PHASE 1/2 TRAIN  (offline, local Pile=${PILE})"
if [[ ! -e "${PILE}" ]] && ! compgen -G "${PILE}" >/dev/null; then
  echo "[fp8te-lrbs] WARNING: local Pile '${PILE}' not found — training would hit HF. Aborting."
  exit 1
fi
LOCAL_DATA="${PILE}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PHASE=train bash "${SD}/sweep_lr_batch.sh"

echo "[fp8te-lrbs] $(date)  PHASE 2/2 EVAL  (online; github-code streams from HF)"
PHASE=eval bash "${SD}/sweep_lr_batch.sh"

echo "[fp8te-lrbs] $(date)  pipeline complete (variant=${RUN_GROUP:-raw})."
