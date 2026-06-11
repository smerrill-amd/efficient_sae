#!/usr/bin/env bash
# ghost_pipeline.sh — wait for the RAW lr×batch sessions on this GPU to finish,
# then run the GHOST (grouped BatchTopK) version of the SAME grid: train + eval.
#
# "Ghost batch" = BatchTopK top-k taken within fixed-size groups (GROUP_SIZE) instead
# of the whole batch (architectures/grouped_batchtopk.py / the fp8 arch's group option),
# so batch size becomes a PURE optimization knob and can be compared cell-for-cell
# against the raw grid to disentangle optimization dynamics from BatchTopK selection.
#
# Queued (not parallel) so it only starts once the raw train+eval free this GPU.
#
# Required env:
#   WAIT_SESSIONS   space-separated tmux sessions to wait for before starting
#                   (e.g. "lrbs_fp16 lrbs_fp16_eval")
# Plus the usual sweep_lr_batch.sh knobs (MODEL, PRECISIONS, WIDTH, K, LRS, BATCHES,
#   EVALS, CHECKPOINTS, GPU). GROUP_SIZE/RUN_GROUP/PHASE are forced below.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SESSIONS="${WAIT_SESSIONS:-}"
POLL="${POLL:-60}"

echo "[ghost] $(date)  will wait for sessions: [${WAIT_SESSIONS}] (poll ${POLL}s)"
for s in ${WAIT_SESSIONS}; do
  echo "[ghost] waiting for tmux session '${s}' ..."
  while tmux has-session -t "${s}" 2>/dev/null; do sleep "${POLL}"; done
  echo "[ghost] '${s}' has ended at $(date)."
done

# Clear any stale ghost dirs for this precision so native-100M starts fresh
# (a leftover partial run would otherwise trigger a resume with a mismatched cfg).
_sfx=""; [[ "${PRECISIONS:-}" == *fp8* ]] && _sfx="_fp8"
rm -rf "${SCRIPT_DIR}/results/saebench_${MODEL:-gemma}${_sfx}_ghost_"* 2>/dev/null && \
  echo "[ghost] cleared stale ghost dirs (saebench_${MODEL:-gemma}${_sfx}_ghost_*)"

echo "[ghost] $(date)  starting GHOST grid (GROUP_SIZE=${GROUP_SIZE:-2048}, RUN_GROUP=ghost,"
echo "[ghost]          PHASE=all, CHECKPOINTS=${CHECKPOINTS:-all}) on GPU ${GPU:-?}"
GROUP_SIZE="${GROUP_SIZE:-2048}" RUN_GROUP=ghost PHASE=all \
  "${SCRIPT_DIR}/sweep_lr_batch.sh"
echo "[ghost] $(date)  ghost pipeline (train + eval) complete."
