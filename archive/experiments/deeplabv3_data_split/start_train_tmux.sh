#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/logs/train_tmux.log"
SESSION_NAME="${SESSION_NAME:-deeplabv3_data_split}"
TMUX_SOCKET="${TMUX_SOCKET:-deeplabv3_data_split}"
DEVICES="${DEVICES:-1}"
TRAIN_ARGS="${TRAIN_ARGS:---device gpu --train-batch-size 2 --eval-batch-size 2 --disable-backbone-pretrained}"

mkdir -p "${OUTPUT_DIR}/logs"

if command -v tmux >/dev/null 2>&1; then
  if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "session already running: ${SESSION_NAME}"
    exit 0
  fi

  tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION_NAME}" -c <PROJECT_ROOT> \
    "bash -lc 'source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && export FLAGS_allocator_strategy=auto_growth && export CUDA_VISIBLE_DEVICES=${DEVICES} && cd <PROJECT_ROOT> && python -u <PROJECT_ROOT>/paddlers/deeplabv3_data_split/train.py --run-train ${TRAIN_ARGS} >> ${LOG_FILE} 2>&1'"
  echo "prepared tmux launch command for ${SESSION_NAME}"
  exit 0
fi

echo "tmux is not available on this host"
exit 1
