#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/factseglog"
SESSION_NAME="fastseg_gid15_nromal"
TMUX_SOCKET="fastseg_gid15_nromal"
DEVICE_ID="${DEVICE_ID:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"

mkdir -p "${OUTPUT_DIR}"

source ${HOME}/miniconda3/etc/profile.d/conda.sh
set +u
conda activate paddlers
set -u
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"

if command -v tmux >/dev/null 2>&1; then
  if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "session already running: ${SESSION_NAME}"
    exit 0
  fi

  tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION_NAME}" -c <PROJECT_ROOT> \
    "bash -lc 'source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && export CUDA_VISIBLE_DEVICES=${DEVICE_ID} && cd <PROJECT_ROOT> && python -u <PROJECT_ROOT>/paddlers/fastseg_gid15_nromal/train.py --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} --learning-rate ${LEARNING_RATE} >> ${LOG_FILE} 2>&1'"
  sleep 2
  if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "started in tmux session ${SESSION_NAME}"
    exit 0
  fi
fi

nohup bash -lc "source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && export CUDA_VISIBLE_DEVICES=${DEVICE_ID} && cd <PROJECT_ROOT> && python -u <PROJECT_ROOT>/paddlers/fastseg_gid15_nromal/train.py --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} --learning-rate ${LEARNING_RATE} >> ${LOG_FILE} 2>&1" >/dev/null 2>&1 &
sleep 2
echo "started in nohup mode"
