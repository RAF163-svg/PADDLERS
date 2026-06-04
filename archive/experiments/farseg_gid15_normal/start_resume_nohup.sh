#!/usr/bin/env bash

# Start GID-15 resume training in a detached nohup + setsid process.
# PID file:
#   <PROJECT_ROOT>/paddlers/farseg_gid15_normal/output/train_resume.pid
# Log file:
#   <PROJECT_ROOT>/paddlers/farseg_gid15_normal/output/train_resume_from_epoch5.log
# Usage:
#   bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/start_resume_nohup.sh
#   CUDA_DEVICE=1 bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/start_resume_nohup.sh

set -eo pipefail

REPO_ROOT="<PROJECT_ROOT>"
EXP_DIR="${REPO_ROOT}/paddlers/farseg_gid15_normal"
OUTPUT_DIR="${EXP_DIR}/output"
RESUME_SCRIPT="${EXP_DIR}/resume_from_epoch5.sh"
LOG_FILE="${OUTPUT_DIR}/train_resume_from_epoch5.log"
PID_FILE="${OUTPUT_DIR}/train_resume.pid"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${EXP_DIR}"

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is not installed; cannot start detached nohup launcher." >&2
  exit 1
fi

if ! CHECKPOINT_PATH="$("${RESUME_SCRIPT}" --print-checkpoint "${1:-}")"; then
  echo "failed to resolve resume checkpoint" >&2
  exit 1
fi

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate paddlers

emit_preflight() {
  printf '\n========== Resume Launch (%s) ==========\n' "$(date '+%F %T %z')"
  printf '[launcher] mode=nohup\n'
  printf '[launcher] host=%s\n' "$(hostname)"
  printf '[launcher] work_dir=%s\n' "$(pwd)"
  printf '[launcher] python=%s\n' "$(command -v python)"
  printf '[launcher] conda_env=%s\n' "${CONDA_DEFAULT_ENV:-unknown}"
  printf '[launcher] cuda_visible_devices=%s\n' "${CUDA_DEVICE}"
  printf '[launcher] checkpoint_path=%s\n' "${CHECKPOINT_PATH}"
  printf '[launcher] log_path=%s\n' "${LOG_FILE}"
}

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && ps -p "${existing_pid}" >/dev/null 2>&1; then
    emit_preflight | tee -a "${LOG_FILE}"
    {
      echo "training process already running: PID ${existing_pid}"
      echo "check process: ps -fp ${existing_pid}"
      echo "tail log: tail -f ${LOG_FILE}"
    } | tee -a "${LOG_FILE}"
    exit 0
  fi
fi

emit_preflight | tee -a "${LOG_FILE}"

nohup setsid /bin/bash -lc \
  "export CUDA_DEVICE=${CUDA_DEVICE}; export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}; export PYTHONUNBUFFERED=1; /bin/bash ${RESUME_SCRIPT} ${CHECKPOINT_PATH}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${PID_FILE}"
sleep 2

if ! ps -p "${pid}" >/dev/null 2>&1; then
  echo "failed to start detached training process; check ${LOG_FILE}" | tee -a "${LOG_FILE}" >&2
  exit 1
fi

{
  echo "started detached training process"
  echo "PID: ${pid}"
  echo "log_path: ${LOG_FILE}"
  echo "check process: ps -fp ${pid}"
  echo "tail log: tail -f ${LOG_FILE}"
} | tee -a "${LOG_FILE}"
