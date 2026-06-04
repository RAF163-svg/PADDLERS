#!/usr/bin/env bash

# Resume GID-15 training from the latest available checkpoint.
# Default log file when used with the launcher scripts:
#   <PROJECT_ROOT>/paddlers/farseg_gid15_normal/output/train_resume_from_epoch5.log
# Usage:
#   bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/resume_from_epoch5.sh
#   bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/resume_from_epoch5.sh /abs/path/to/checkpoint_dir
#   bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/resume_from_epoch5.sh --print-checkpoint

set -eo pipefail

REPO_ROOT="<PROJECT_ROOT>"
EXP_DIR="${REPO_ROOT}/paddlers/farseg_gid15_normal"
OUTPUT_DIR="${EXP_DIR}/output"
TRAIN_SCRIPT="${EXP_DIR}/train.py"
DEFAULT_FALLBACK_CHECKPOINT="${OUTPUT_DIR}/epoch_5"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
LOG_FILE="${OUTPUT_DIR}/train_resume_from_epoch5.log"

resolve_checkpoint_dir() {
  local requested_path="${1:-}"
  local latest_epoch_dir=""

  if [[ -n "${requested_path}" ]]; then
    if [[ -d "${requested_path}" && -f "${requested_path}/model.pdparams" && -f "${requested_path}/model.pdopt" ]]; then
      printf '%s\n' "${requested_path}"
      return 0
    fi
    echo "[resume] invalid checkpoint directory: ${requested_path}" >&2
    return 1
  fi

  latest_epoch_dir="$(
    find "${OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'epoch_*' 2>/dev/null \
      | sort -V \
      | while IFS= read -r epoch_dir; do
          if [[ -f "${epoch_dir}/model.pdparams" && -f "${epoch_dir}/model.pdopt" ]]; then
            printf '%s\n' "${epoch_dir}"
          fi
        done \
      | tail -n 1
  )"
  if [[ -n "${latest_epoch_dir}" ]]; then
    printf '%s\n' "${latest_epoch_dir}"
    return 0
  fi

  if [[ -d "${DEFAULT_FALLBACK_CHECKPOINT}" \
        && -f "${DEFAULT_FALLBACK_CHECKPOINT}/model.pdparams" \
        && -f "${DEFAULT_FALLBACK_CHECKPOINT}/model.pdopt" ]]; then
    printf '%s\n' "${DEFAULT_FALLBACK_CHECKPOINT}"
    return 0
  fi

  if [[ -d "${OUTPUT_DIR}/best_model" \
        && -f "${OUTPUT_DIR}/best_model/model.pdparams" \
        && -f "${OUTPUT_DIR}/best_model/model.pdopt" ]]; then
    printf '%s\n' "${OUTPUT_DIR}/best_model"
    return 0
  fi

  echo "[resume] no resumable checkpoint found under ${OUTPUT_DIR}" >&2
  return 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '1,8p' "${BASH_SOURCE[0]}"
  exit 0
fi

if [[ "${1:-}" == "--print-checkpoint" ]]; then
  resolve_checkpoint_dir "${2:-}"
  exit 0
fi

CHECKPOINT_DIR="$(resolve_checkpoint_dir "${1:-}")"

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate paddlers

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

cd "${EXP_DIR}"
echo "[resume] time=$(date '+%F %T %z')"
echo "[resume] host=$(hostname)"
echo "[resume] work_dir=$(pwd)"
echo "[resume] python=$(command -v python)"
echo "[resume] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[resume] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[resume] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[resume] log_file=${LOG_FILE}"

exec python -u "${TRAIN_SCRIPT}" \
  --resume-checkpoint "${CHECKPOINT_DIR}"
