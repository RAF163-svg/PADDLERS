#!/usr/bin/env bash

# Resume the FastSCNN GID-15 training job from the latest recoverable checkpoint.
# Usage:
#   bash resume_from_latest.sh
#   CUDA_DEVICE=1 bash resume_from_latest.sh
# Logs:
#   output/train_resume.log

set -eo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
OUTPUT_DIR="${WORK_DIR}/output"
TRAIN_SCRIPT="${WORK_DIR}/train.py"
CONFIG_PATH="${WORK_DIR}/train_config.json"
LOG_FILE="${OUTPUT_DIR}/train_resume.log"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

mkdir -p "${OUTPUT_DIR}"

find_latest_checkpoint() {
  local latest=""
  while IFS= read -r path; do
    [[ -f "${path}/model.pdparams" ]] || continue
    [[ -f "${path}/model.pdopt" ]] || continue
    [[ -f "${path}/model.yml" ]] || continue
    latest="${path}"
  done < <(find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'epoch_*' | sort -V)

  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return 0
  fi

  if [[ -f "${OUTPUT_DIR}/best_model/model.pdparams" && -f "${OUTPUT_DIR}/best_model/model.pdopt" && -f "${OUTPUT_DIR}/best_model/model.yml" ]]; then
    printf '%s\n' "${OUTPUT_DIR}/best_model"
    return 0
  fi

  return 1
}

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if ! CHECKPOINT_PATH="$(find_latest_checkpoint)"; then
    echo "no recoverable checkpoint found under ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi

source "${CONDA_SH}"
set +u
conda activate paddlers
set -e

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

{
  echo "==== FastSCNN GID-15 resume launch ===="
  echo "time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "workdir: $(pwd)"
  echo "python: $(command -v python)"
  echo "CUDA_DEVICE: ${CUDA_DEVICE}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
  echo "conda_env: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "checkpoint_path: ${CHECKPOINT_PATH}"
  echo "log_path: ${LOG_FILE}"
  echo "train_script: ${TRAIN_SCRIPT}"
  echo "config_path: ${CONFIG_PATH}"
  echo
  python -u "${TRAIN_SCRIPT}" --config "${CONFIG_PATH}" --resume-checkpoint "${CHECKPOINT_PATH}"
} 2>&1 | tee -a "${LOG_FILE}"
