#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
LOG_DIR="${OUTPUT_DIR}/logs"
PID_FILE="${OUTPUT_DIR}/train.pid"
LAUNCH_LOG="${LOG_DIR}/launcher.stdout"
GPU_ID="${GPU_ID:-1}"

mkdir -p "${LOG_DIR}"

source ${HOME}/miniconda3/etc/profile.d/conda.sh

CMD="cd \"${SCRIPT_DIR}\" && \
conda activate paddlers && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
python train.py --config train_config_multihead.json --run-train --device gpu"

setsid bash -lc "${CMD}" > "${LAUNCH_LOG}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Started 4-head routed HRNet training."
echo "PID=${PID}"
echo "GPU_ID=${GPU_ID}"
echo "launcher_log=${LAUNCH_LOG}"
echo "train_log=${LOG_DIR}/train.log"
echo "eval_log=${LOG_DIR}/eval.log"
