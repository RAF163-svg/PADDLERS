#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/factseglog"
SESSION_NAME="fastseg_gid15_nromal"
TMUX_SOCKET="fastseg_gid15_nromal"

echo "[processes]"
ps -eo pid,cmd | grep -E 'fastseg_gid15_nromal/train.py' | grep -v grep || true

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

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
