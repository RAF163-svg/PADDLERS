#!/usr/bin/env bash

set -eo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output}"
TRAIN_SCRIPT="${WORK_DIR}/train.py"
CONFIG_PATH="${CONFIG_PATH:-${WORK_DIR}/train_config.json}"
LOG_FILE="${OUTPUT_DIR}/train_nohup.log"
PID_FILE="${OUTPUT_DIR}/train.pid"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

quote() {
  printf '%q' "$1"
}

PROJECT_ROOT_Q="$(quote "${PROJECT_ROOT}")"
CONDA_SH_Q="$(quote "${CONDA_SH}")"
TRAIN_SCRIPT_Q="$(quote "${TRAIN_SCRIPT}")"
CONFIG_PATH_Q="$(quote "${CONFIG_PATH}")"
CUDA_DEVICE_Q="$(quote "${CUDA_DEVICE}")"

nohup setsid bash -lc "source ${CONDA_SH_Q} && set +u && conda activate paddlers && set -e && cd ${PROJECT_ROOT_Q} && export PYTHONUNBUFFERED=1 && export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_Q} && { echo '==== HRNet GID-15 nohup launch ===='; echo \"time: \$(date)\"; echo \"host: \$(hostname)\"; echo \"workdir: \$(pwd)\"; echo \"python: \$(command -v python)\"; echo \"CUDA_DEVICE: ${CUDA_DEVICE}\"; echo \"CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-}\"; echo \"conda_env: \${CONDA_DEFAULT_ENV:-unknown}\"; echo 'checkpoint_path: None'; echo 'log_path: ${LOG_FILE}'; echo 'train_script: ${TRAIN_SCRIPT}'; echo 'config_path: ${CONFIG_PATH}'; echo; python -u ${TRAIN_SCRIPT_Q} --config ${CONFIG_PATH_Q}; }" >"${LOG_FILE}" 2>&1 < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${PID_FILE}"

echo "pid: ${PID}"
echo "log_path: ${LOG_FILE}"
echo "tail: tail -f ${LOG_FILE}"
echo "ps: ps -fp ${PID}"
