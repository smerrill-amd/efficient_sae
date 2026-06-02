#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./shell_scripts/setup_rocm_sae_env.sh
#
# Run this from inside the pre-built ROCm container. It installs the Python
# dependencies, validates the ROCm/SAELens stack, configures the Jupyter kernel
# env, and launches JupyterLab.

# Resolve the project root from this script's location (shell_scripts/ -> project root)
# so the script works regardless of where the repo is checked out. Override by
# exporting PROJECT_ROOT before invoking.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

REQ_FILE="${REQ_FILE:-${PROJECT_ROOT}/requirements.txt}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements.txt not found at ${REQ_FILE}"
  exit 1
fi

echo "Installing python dependencies from ${REQ_FILE}..."
"${PYTHON}" -m pip install -r "${REQ_FILE}"

echo "Validating ROCm and SAELens..."
"${PYTHON}" -c "import torch, sae_lens; print('torch:', torch.__version__); print('sae_lens:', sae_lens.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count())"

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
# Default to GPU 0 so kernel restarts take ~5s instead of ~60s (ROCm inits all visible GPUs).
# Override: JUPYTER_GPU=0,1,2,3 ./setup_rocm_sae_env.sh
# Or inside a notebook before importing torch:
#   import os; os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"; import torch
JUPYTER_GPU="${JUPYTER_GPU:-0}"

echo "Stopping any existing Jupyter servers..."
pkill -f jupyter 2>/dev/null && sleep 2 || true

# Bake HIP_VISIBLE_DEVICES into the Python kernel spec so every spawned kernel
# inherits the GPU limit — this is what actually fixes slow kernel restarts.
echo "Configuring kernel env (HIP_VISIBLE_DEVICES=${JUPYTER_GPU})..."
JUPYTER_GPU="${JUPYTER_GPU}" "${PYTHON}" -c "
import json, os, sys

gpu = os.environ['JUPYTER_GPU']

kernel_dirs = [
    '/opt/venv/share/jupyter/kernels/python3',
    os.path.expanduser('~/.local/share/jupyter/kernels/python3'),
]

spec_template = {
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3 (ROCm)',
    'language': 'python',
    'env': {}
}

for kdir in kernel_dirs:
    os.makedirs(kdir, exist_ok=True)
    kfile = os.path.join(kdir, 'kernel.json')
    try:
        with open(kfile) as f:
            spec = json.load(f)
    except Exception:
        spec = dict(spec_template)
    spec.setdefault('env', {})
    spec['env']['HIP_VISIBLE_DEVICES'] = gpu
    spec['env']['ROCR_VISIBLE_DEVICES'] = gpu
    spec['display_name'] = 'Python 3 (ROCm GPU %s)' % gpu
    with open(kfile, 'w') as f:
        json.dump(spec, f, indent=2)
    print('Updated:', kfile)
"

echo "Starting JupyterLab on port ${JUPYTER_PORT}..."
HIP_VISIBLE_DEVICES="${JUPYTER_GPU}" ROCR_VISIBLE_DEVICES="${JUPYTER_GPU}" \
  nohup jupyter lab \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT}" \
    --no-browser \
    --allow-root \
    --notebook-dir="${PROJECT_ROOT}" \
    --ServerApp.token='' \
    --ServerApp.password='' \
    >/tmp/jupyterlab.log 2>&1 &

echo -n "Waiting for Jupyter to start"
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${JUPYTER_PORT}/api" > /dev/null 2>&1; then
    echo " ready."
    break
  fi
  echo -n "."
  sleep 1
done

echo
echo "========================================"
echo "  JupyterLab:  http://localhost:${JUPYTER_PORT}"
echo "  GPU(s):      HIP_VISIBLE_DEVICES=${JUPYTER_GPU}"
echo "  Log:         /tmp/jupyterlab.log"
echo "========================================"
echo
echo "In Cursor: open a .ipynb → Select Kernel → Existing Jupyter Server → http://localhost:${JUPYTER_PORT}"
