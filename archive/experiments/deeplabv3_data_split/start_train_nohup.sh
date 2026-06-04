#!/usr/bin/env bash

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/logs/train_nohup.log"
DEVICES="${DEVICES:-1}"
TRAIN_ARGS="${TRAIN_ARGS:---device gpu --train-batch-size 2 --eval-batch-size 2 --disable-backbone-pretrained}"

mkdir -p "${OUTPUT_DIR}/logs"

nohup bash -lc "source ${HOME}/miniconda3/etc/profile.d/conda.sh && set +u && conda activate paddlers && set -u && export PYTHONUNBUFFERED=1 && export FLAGS_allocator_strategy=auto_growth && export CUDA_VISIBLE_DEVICES=${DEVICES} && cd <PROJECT_ROOT> && python -u <PROJECT_ROOT>/paddlers/deeplabv3_data_split/train.py --run-train ${TRAIN_ARGS} >> ${LOG_FILE} 2>&1" >/dev/null 2>&1 &
echo "prepared nohup launch command"
