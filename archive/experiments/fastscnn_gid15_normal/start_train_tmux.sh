#!/usr/bin/env bash

set -eo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
TRAIN_SCRIPT="${WORK_DIR}/train.py"
CONFIG_PATH="${WORK_DIR}/train_config.json"
LOG_FILE="${OUTPUT_DIR}/train_tmux.log"
SESSION_NAME="fastscnn_gid15_train"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please install tmux first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach: tmux attach -t ${SESSION_NAME}"
  echo "logs: tail -f ${LOG_FILE}"
  echo "sessions: tmux ls"
  exit 0
fi

quote() {
  printf '%q' "$1"
}

PROJECT_ROOT_Q="$(quote "${PROJECT_ROOT}")"
CONDA_SH_Q="$(quote "${CONDA_SH}")"
TRAIN_SCRIPT_Q="$(quote "${TRAIN_SCRIPT}")"
CONFIG_PATH_Q="$(quote "${CONFIG_PATH}")"
LOG_FILE_Q="$(quote "${LOG_FILE}")"
CUDA_DEVICE_Q="$(quote "${CUDA_DEVICE}")"

tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" \
  "bash -lc 'source ${CONDA_SH_Q} && set +u && conda activate paddlers && set -e && cd ${PROJECT_ROOT_Q} && export PYTHONUNBUFFERED=1 && export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_Q} && { echo \"==== FastSCNN GID-15 tmux launch ====\"; echo \"time: \$(date)\"; echo \"host: \$(hostname)\"; echo \"workdir: \$(pwd)\"; echo \"python: \$(command -v python)\"; echo \"CUDA_DEVICE: ${CUDA_DEVICE}\"; echo \"CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-}\"; echo \"conda_env: \${CONDA_DEFAULT_ENV:-unknown}\"; echo \"checkpoint_path: None\"; echo \"log_path: ${LOG_FILE}\"; echo \"train_script: ${TRAIN_SCRIPT}\"; echo \"config_path: ${CONFIG_PATH}\"; echo; python -u ${TRAIN_SCRIPT_Q} --config ${CONFIG_PATH_Q}; } 2>&1 | tee -a ${LOG_FILE_Q}'"

echo "tmux session started: ${SESSION_NAME}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "logs: tail -f ${LOG_FILE}"
echo "sessions: tmux ls"
