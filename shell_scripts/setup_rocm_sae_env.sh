#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./shell_scripts/setup_rocm_sae_env.sh [optional-image-tar]
#
# Examples:
#   ./shell_scripts/setup_rocm_sae_env.sh
#   ./shell_scripts/setup_rocm_sae_env.sh /path/to/rocm_pytorch_image.tar

IMAGE_NAME="${IMAGE_NAME:-rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1}"
CONTAINER_NAME="${CONTAINER_NAME:-sm_image}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/smerrill@amd.com/efficient_sae}"
REQ_FILE="${REQ_FILE:-${PROJECT_ROOT}/requirements.txt}"
IMAGE_TAR_PATH="${1:-${DOCKER_IMAGE_TAR:-}}"

if [[ -n "${IMAGE_TAR_PATH}" ]]; then
  echo "Loading docker image tar: ${IMAGE_TAR_PATH}"
  docker load -i "${IMAGE_TAR_PATH}"
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements.txt not found at ${REQ_FILE}"
  exit 1
fi

if [[ "$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')" == "${CONTAINER_NAME}" ]]; then
  echo "Container ${CONTAINER_NAME} already running."
elif [[ "$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')" == "${CONTAINER_NAME}" ]]; then
  echo "Starting existing container ${CONTAINER_NAME}."
  docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "Starting new ROCm container ${CONTAINER_NAME} from ${IMAGE_NAME}."
  docker run -d \
    --rm \
    --network=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 200G \
    -v /home:/home \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}" \
    sleep infinity >/dev/null
fi

echo "Installing python dependencies from ${REQ_FILE} in ${CONTAINER_NAME}..."
docker exec "${CONTAINER_NAME}" python3 -m pip install -r "${REQ_FILE}"

echo "Validating ROCm and SAELens inside container..."
docker exec "${CONTAINER_NAME}" python3 -c "import torch, sae_lens; print('torch:', torch.__version__); print('sae_lens:', sae_lens.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count())"

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
# Default to GPU 0 so kernel restarts take ~5s instead of ~60s (ROCm inits all visible GPUs).
# Override: JUPYTER_GPU=0,1,2,3 ./setup_rocm_sae_env.sh
# Or inside a notebook before importing torch:
#   import os; os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"; import torch
JUPYTER_GPU="${JUPYTER_GPU:-0}"

echo "Stopping any existing Jupyter servers in ${CONTAINER_NAME}..."
docker exec "${CONTAINER_NAME}" bash -c "pkill -f jupyter 2>/dev/null; sleep 2; true" || true

# Bake HIP_VISIBLE_DEVICES into the Python kernel spec so every spawned kernel
# inherits the GPU limit — this is what actually fixes slow kernel restarts.
echo "Configuring kernel env (HIP_VISIBLE_DEVICES=${JUPYTER_GPU})..."
docker exec "${CONTAINER_NAME}" python3 -c "
import json, os, sys

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
    spec['env']['HIP_VISIBLE_DEVICES'] = '${JUPYTER_GPU}'
    spec['env']['ROCR_VISIBLE_DEVICES'] = '${JUPYTER_GPU}'
    spec['display_name'] = 'Python 3 (ROCm GPU ${JUPYTER_GPU})'
    with open(kfile, 'w') as f:
        json.dump(spec, f, indent=2)
    print('Updated:', kfile)
"

echo "Starting JupyterLab on port ${JUPYTER_PORT}..."
docker exec -d "${CONTAINER_NAME}" bash -c "
  export HIP_VISIBLE_DEVICES=${JUPYTER_GPU}
  export ROCR_VISIBLE_DEVICES=${JUPYTER_GPU}
  jupyter lab \
    --ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --no-browser \
    --allow-root \
    --notebook-dir='${PROJECT_ROOT}' \
    --ServerApp.token='' \
    --ServerApp.password=''
"

echo -n "Waiting for Jupyter to start"
for i in $(seq 1 30); do
  if docker exec "${CONTAINER_NAME}" bash -c "curl -sf http://localhost:${JUPYTER_PORT}/api > /dev/null 2>&1"; then
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
echo "========================================"
echo
echo "In Cursor: open a .ipynb → Select Kernel → Existing Jupyter Server → http://localhost:${JUPYTER_PORT}"
echo "Open shell: docker exec -it ${CONTAINER_NAME} /bin/bash"
