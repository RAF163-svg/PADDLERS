#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 CONFIG_PATH SAVE_DIR [extra train.py args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train.py}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-paddlers}"
CONFIG_PATH="$1"
SAVE_DIR="$2"
shift 2

CUDA_DEVICE="${CUDA_DEVICE:-1}"
PID_FILE="${SAVE_DIR}/train.pid"

mkdir -p "${SAVE_DIR}/logs"

printf -v CONFIG_PATH_Q '%q' "${CONFIG_PATH}"
printf -v SAVE_DIR_Q '%q' "${SAVE_DIR}"
printf -v TRAIN_SCRIPT_Q '%q' "${TRAIN_SCRIPT}"
printf -v PROJECT_ROOT_Q '%q' "${PROJECT_ROOT}"
printf -v CUDA_DEVICE_Q '%q' "${CUDA_DEVICE}"

EXTRA_ARGS_Q=""
if [ "$#" -gt 0 ]; then
  printf -v EXTRA_ARGS_Q '%q ' "$@"
fi

nohup setsid bash -lc "source ${CONDA_SH@Q} && conda activate ${CONDA_ENV@Q} && export PROJECT_ROOT=${PROJECT_ROOT_Q} && export PYTHONUNBUFFERED=1 && export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_Q} && export FLAGS_allocator_strategy=auto_growth && export GLOG_minloglevel=2 && cd ${PROJECT_ROOT_Q} && python -u ${TRAIN_SCRIPT_Q} --config ${CONFIG_PATH_Q} --save-dir ${SAVE_DIR_Q} --device gpu --run-train ${EXTRA_ARGS_Q}" >/dev/null 2>&1 < /dev/null &

echo $! > "${PID_FILE}"
echo "started_pid=$(cat "${PID_FILE}")"
echo "pid_file=${PID_FILE}"
