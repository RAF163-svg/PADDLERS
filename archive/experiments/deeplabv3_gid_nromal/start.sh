#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/deeplabv3log"
SESSION_NAME="deeplabv3_gid_nromal"
TMUX_SOCKET="deeplabv3_gid_nromal"
DEVICES="${DEVICES:-0,1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
DIST_LOG_DIR="${OUTPUT_DIR}/distributed_logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DIST_LOG_DIR}"

source ${HOME}/miniconda3/etc/profile.d/conda.sh
set +u
conda activate paddlers
set -u
export PYTHONUNBUFFERED=1
PADDLE_LIB_DIR="$(python - <<'PY'
import pathlib
import paddle
print(pathlib.Path(paddle.__file__).resolve().parent / "libs")
PY
)"
export LD_LIBRARY_PATH="${PADDLE_LIB_DIR}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if command -v tmux >/dev/null 2>&1; then
  if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "session already running: ${SESSION_NAME}"
    exit 0
  fi

  tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION_NAME}" -c <PROJECT_ROOT> \
    "bash -lc 'source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && cd <PROJECT_ROOT> && python -u -m paddle.distributed.launch --devices ${DEVICES} --log_dir ${DIST_LOG_DIR} <PROJECT_ROOT>/paddlers/deeplabv3_gid_nromal/train.py --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} >> ${LOG_FILE} 2>&1'"
  sleep 2
  if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "started in tmux session ${SESSION_NAME}"
    exit 0
  fi
fi

nohup bash -lc "source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && cd <PROJECT_ROOT> && python -u -m paddle.distributed.launch --devices ${DEVICES} --log_dir ${DIST_LOG_DIR} <PROJECT_ROOT>/paddlers/deeplabv3_gid_nromal/train.py --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} >> ${LOG_FILE} 2>&1" >/dev/null 2>&1 &
sleep 2
echo "started in nohup mode"
