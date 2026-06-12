#!/usr/bin/env bash
# repro_experiment.sh — reproducibility & seed-variance study for our SAEBench SAEs.
#
# Trains the SAME Pythia SAE config many times and lets the companion notebook
# (notebooks/reproducibility_analysis.ipynb) quantify how much the result depends
# on the RNG seed vs. is fixed once the seed is pinned. Two parts:
#
#   PART A — DETERMINISM (same seed twice)
#     Two independent training PROCESSES with the SAME seed (default 0) and
#     --deterministic. If our seeding works end-to-end the two SAEs should be
#     bit-identical (weights) and give identical SAEBench evals.
#       runs:  saebench_<model>_repro_seed0        (== variance seed 0, reused)
#              saebench_<model>_repro_seed0_dup     (the duplicate process)
#
#   PART B — SEED VARIANCE (N different seeds)
#     N independent runs with seeds 0..N-1. The notebook measures the spread of
#     every eval metric and of pairwise weight similarity across seeds — i.e. how
#     much "luck of the seed" is baked into a single SAE.
#       runs:  saebench_<model>_repro_seed{0..N-1}
#
# Every run trains ONE member (default w16384_k40) for TRAIN_TOKENS (default 100M)
# and is then evaluated with the core + sparse_probing SAEBench suites.
#
# Env knobs (all overridable):
#   MODEL=pythia  WIDTH=16384  K=40  TRAIN_TOKENS=100000000  NSEEDS=10
#   GPU=0  DETERMINISTIC=1  EVALS=core,sparse_probing
#   SKIP_TRAIN=0  SKIP_EVAL=0  DRY_RUN=0
#
# Examples:
#   GPU=1 ./repro_experiment.sh                       # full study on GPU 1
#   NSEEDS=4 WIDTH=4096 ./repro_experiment.sh         # quick/cheap variant
#   DRY_RUN=1 ./repro_experiment.sh                   # print the plan, no work
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TRAIN="${SCRIPT_DIR}/train_saebench_replication.py"
EVAL_SH="${SCRIPT_DIR}/eval_saebench.sh"
OUT="${OUT:-${SCRIPT_DIR}/results}"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# cuBLAS needs this set BEFORE the process starts for deterministic GEMMs.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# ── Knobs ─────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-pythia}"
WIDTH="${WIDTH:-16384}"
K="${K:-40}"
MEMBER="w${WIDTH}_k${K}"
TRAIN_TOKENS="${TRAIN_TOKENS:-100000000}"
NSEEDS="${NSEEDS:-10}"
GPU="${GPU:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
EVALS="${EVALS:-core,sparse_probing}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

det_flag=(); [[ "${DETERMINISTIC}" == "1" ]] && det_flag=( --deterministic )
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

echo "============================================================"
echo "  Reproducibility / seed-variance study"
echo "  model=${MODEL}  member=${MEMBER}  tokens=${TRAIN_TOKENS}"
echo "  seeds=0..$((NSEEDS-1))  + seed0_dup   deterministic=${DETERMINISTIC}"
echo "  evals=[${EVALS}]  gpu=${GPU}  out=${OUT}"
echo "============================================================"

# Train one run for a given seed + run-tag (idempotent: skips if final SAE exists).
train_one() {  # $1=seed  $2=tag
  local seed="$1" tag="$2"
  local dir="${OUT}/saebench_${MODEL}_${tag}"
  if [[ -f "${dir}/${MEMBER}/cfg.json" ]]; then
    echo "[repro] TRAIN skip ${tag} (final SAE already at ${dir}/${MEMBER})"
    return 0
  fi
  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo "[repro] TRAIN skip ${tag} (SKIP_TRAIN=1, no SAE yet)"; return 0
  fi
  local log="${OUT}/repro_train_${MODEL}_${tag}_${TS}.log"
  echo "[repro] TRAIN ${tag}  seed=${seed}  -> ${dir}  (log ${log})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "        ${PYTHON} ${TRAIN} --model ${MODEL} --gpu ${GPU} --widths ${WIDTH}" \
         "--ks ${K} --training-tokens ${TRAIN_TOKENS} --seed ${seed} ${det_flag[*]}" \
         "--run-tag ${tag} --no-wandb --output-dir ${OUT}"
    return 0
  fi
  "${PYTHON}" "${TRAIN}" --model "${MODEL}" --gpu "${GPU}" \
     --widths "${WIDTH}" --ks "${K}" --training-tokens "${TRAIN_TOKENS}" \
     --seed "${seed}" "${det_flag[@]}" --run-tag "${tag}" \
     --no-wandb --output-dir "${OUT}" > "${log}" 2>&1
  if [[ $? -ne 0 ]]; then echo "[repro] TRAIN FAILED ${tag} (see ${log})"; return 1; fi
}

# Eval one run's final SAE (idempotent unless FORCE_RERUN=1 is exported).
eval_one() {  # $1=tag
  local tag="$1"
  local dir="${OUT}/saebench_${MODEL}_${tag}"
  if [[ ! -f "${dir}/${MEMBER}/cfg.json" ]]; then
    echo "[repro] EVAL skip ${tag} (no final SAE)"; return 0
  fi
  if [[ "${SKIP_EVAL}" == "1" ]]; then echo "[repro] EVAL skip ${tag} (SKIP_EVAL=1)"; return 0; fi
  echo "[repro] EVAL ${tag}  [${EVALS}]"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "        RUN_DIR=${dir} MEMBER=${MEMBER} CHECKPOINTS=final EVALS=${EVALS} GPU=${GPU} bash ${EVAL_SH}"
    return 0
  fi
  RUN_DIR="${dir}" MEMBER="${MEMBER}" CHECKPOINTS=final EVALS="${EVALS}" GPU="${GPU}" \
    bash "${EVAL_SH}" >> "${OUT}/repro_eval_${MODEL}_${TS}.log" 2>&1 \
    || echo "[repro] EVAL FAILED ${tag} (see ${OUT}/repro_eval_${MODEL}_${TS}.log)"
}

# ── PART B (variance set) + PART A duplicate ──────────────────────────────────
# seeds 0..N-1 are the variance set; seed 0's run is reused as determinism run A.
for s in $(seq 0 $((NSEEDS-1))); do
  train_one "${s}" "repro_seed${s}"
done
# Determinism run B: a second independent process at seed 0.
train_one 0 "repro_seed0_dup"

# ── Evals for every run ───────────────────────────────────────────────────────
for s in $(seq 0 $((NSEEDS-1))); do
  eval_one "repro_seed${s}"
done
eval_one "repro_seed0_dup"

echo "============================================================"
echo "[repro] $(date)  done."
echo "  Determinism pair : saebench_${MODEL}_repro_seed0  vs  saebench_${MODEL}_repro_seed0_dup"
echo "  Variance set     : saebench_${MODEL}_repro_seed{0..$((NSEEDS-1))}"
echo "  Analyse with     : notebooks/reproducibility_analysis.ipynb"
echo "============================================================"
