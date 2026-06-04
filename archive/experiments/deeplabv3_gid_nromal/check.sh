#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/deeplabv3log"
SESSION_NAME="deeplabv3_gid_nromal"
TMUX_SOCKET="deeplabv3_gid_nromal"
DIST_LOG_DIR="${OUTPUT_DIR}/distributed_logs"

echo "[processes]"
ps -eo pid,cmd | grep -E 'deeplabv3_gid_nromal/train.py' | grep -v grep || true

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "[dist_logs]"
if [[ -d "${DIST_LOG_DIR}" ]]; then
  find "${DIST_LOG_DIR}" -maxdepth 2 -type f | sort
else
  echo "distributed log dir not found: ${DIST_LOG_DIR}"
fi

echo
echo "[tmux]"
tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null && \
  tmux -L "${TMUX_SOCKET}" capture-pane -pt "${SESSION_NAME}" | tail -n 20 || \
  echo "tmux session not running"

echo
echo "[train_config]"
if [[ -f "${OUTPUT_DIR}/train_config.json" ]]; then
  cat "${OUTPUT_DIR}/train_config.json"
else
  echo "train_config.json not found"
fi

echo
echo "[log_tail]"
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 40 "${LOG_FILE}"
else
  echo "log file not found: ${LOG_FILE}"
fi

if [[ -f "${DIST_LOG_DIR}/workerlog.0" ]]; then
  echo
  echo "[workerlog.0]"
  tail -n 30 "${DIST_LOG_DIR}/workerlog.0"
fi

if [[ -f "${DIST_LOG_DIR}/workerlog.1" ]]; then
  echo
  echo "[workerlog.1]"
  tail -n 30 "${DIST_LOG_DIR}/workerlog.1"
fi
