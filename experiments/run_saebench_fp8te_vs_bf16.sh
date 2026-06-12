#!/usr/bin/env bash
# run_saebench_fp8te_vs_bf16.sh — end-to-end queue for the trainer_2 comparison.
#
# Replicates the Neuronpedia/SAEBench config you flagged,
#   gemma-2-2b/12-sae_bench_0125-batch_topk-res-64k__trainer_2
# (verified against the published sae_cfg_dict): gemma-2-2b, layer 12 resid_post,
# 65k width, BatchTopK k=80, 500M tokens, lr 3e-4, aux 1/32, threshold beta 0.999,
# seed 0. We train TWO SAEs with byte-for-byte identical hyperparameters and only
# the encoder/decoder GEMM precision differs:
#   1) BF16 baseline           -> results/saebench_gemma
#   2) FP8 TransformerEngine   -> results/saebench_gemma_fp8te
# After each train we run the SAEBench eval suite on member w65536_k80, so the
# companion notebook (notebooks/saebench_compare.ipynb) can compare both to the
# authors' published trainer_2 numbers.
#
# Runs SEQUENTIALLY on ONE GPU (this box has a single MI300X). A failed train
# skips its own eval but does NOT block the other precision.
#
# Usage:   ./run_saebench_fp8te_vs_bf16.sh
# Knobs:   GPU=0  EVALS=core,sparse_probing  (defaults shown)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load secrets (.env): gemma-2-2b is gated -> needs HF token; W&B key optional.
ENV_FILE="${SCRIPT_DIR}/../.env"
[[ -f "${ENV_FILE}" ]] && { set -a; source "${ENV_FILE}"; set +a; }
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-0}"
EVALS="${EVALS:-core,sparse_probing}"   # the headline metrics the notebook reads
MEMBER="w65536_k80"                      # trainer_2 (achieved L0 ~84)
COMMON=(--model gemma --widths 65536 --ks 80 --seed 0 --gpu "${GPU}")

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
banner() { echo; echo "================ $(stamp)  $*  ================"; }

train_then_eval() {  # $1=label  $2=run_dir  ${@:3}=extra train flags
  local label="$1" run_dir="$2"; shift 2
  banner "TRAIN ${label}  -> ${run_dir}"
  if python3 train_saebench_replication.py "${COMMON[@]}" "$@"; then
    banner "EVAL ${label}  (evals=${EVALS})"
    RUN_DIR="${run_dir}" MEMBER="${MEMBER}" EVALS="${EVALS}" GPU="${GPU}" \
      ./eval_saebench.sh || echo ">>> ${label} EVAL FAILED (continuing)"
  else
    echo ">>> ${label} TRAIN FAILED — skipping its eval (continuing to next precision)"
  fi
}

banner "QUEUE START — FP8 TE vs BF16 vs SAEBench trainer_2 (gpu ${GPU})"

# 1) BF16 baseline.
train_then_eval "BF16" "results/saebench_gemma"

# 2) FP8 TransformerEngine (only the GEMM precision changes).
train_then_eval "FP8-TE" "results/saebench_gemma_fp8te" --fp8-te

banner "QUEUE DONE — analyse with notebooks/saebench_compare.ipynb"
