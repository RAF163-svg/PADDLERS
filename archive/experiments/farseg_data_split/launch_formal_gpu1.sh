#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 SAVE_DIR RUN_NAME [extra train.py args...]" >&2
  exit 2
fi

SAVE_DIR="$1"
RUN_NAME="$2"
shift 2

PIDFILE="${SAVE_DIR}/train.pid"

mkdir -p "${SAVE_DIR}"
echo "$$" > "${PIDFILE}"

export CUDA_VISIBLE_DEVICES=1
export FLAGS_allocator_strategy=auto_growth
export MPLCONFIGDIR=/tmp/mpl_farseg_probe

exec ${HOME}/miniconda3/envs/paddlers/bin/python \
  <PROJECT_ROOT>/paddlers/farseg_data_split/train.py \
  --run-train \
  --device gpu \
  --config <PROJECT_ROOT>/paddlers/farseg_data_split/train_config_multihead.json \
  --save-dir "${SAVE_DIR}" \
  --run-name "${RUN_NAME}" \
  "$@"
