#!/usr/bin/env bash

# Start GID-15 resume training inside a persistent tmux session.
# Session name:
#   farseg_gid15_resume
# Log file:
#   <PROJECT_ROOT>/paddlers/farseg_gid15_normal/output/train_resume_from_epoch5.log
# Usage:
#   bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/start_resume_tmux.sh
#   CUDA_DEVICE=1 bash <PROJECT_ROOT>/paddlers/farseg_gid15_normal/start_resume_tmux.sh

set -eo pipefail

REPO_ROOT="<PROJECT_ROOT>"
EXP_DIR="${REPO_ROOT}/paddlers/farseg_gid15_normal"
OUTPUT_DIR="${EXP_DIR}/output"
RESUME_SCRIPT="${EXP_DIR}/resume_from_epoch5.sh"
LOG_FILE="${OUTPUT_DIR}/train_resume_from_epoch5.log"
SESSION_NAME="farseg_gid15_resume"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${EXP_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please use ${EXP_DIR}/start_resume_nohup.sh instead." >&2
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
  printf '[launcher] mode=tmux\n'
  printf '[launcher] host=%s\n' "$(hostname)"
  printf '[launcher] work_dir=%s\n' "$(pwd)"
  printf '[launcher] python=%s\n' "$(command -v python)"
  printf '[launcher] conda_env=%s\n' "${CONDA_DEFAULT_ENV:-unknown}"
  printf '[launcher] cuda_visible_devices=%s\n' "${CUDA_DEVICE}"
  printf '[launcher] checkpoint_path=%s\n' "${CHECKPOINT_PATH}"
  printf '[launcher] log_path=%s\n' "${LOG_FILE}"
}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  emit_preflight | tee -a "${LOG_FILE}"
  {
    echo "tmux session already exists: ${SESSION_NAME}"
    echo "view sessions: tmux list-sessions"
    echo "attach: tmux attach -t ${SESSION_NAME}"
    echo "tail log: tail -f ${LOG_FILE}"
  } | tee -a "${LOG_FILE}"
  exit 0
fi

emit_preflight | tee -a "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" -c "${EXP_DIR}" \
  "/bin/bash -lc 'export CUDA_DEVICE=${CUDA_DEVICE}; export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}; export PYTHONUNBUFFERED=1; /bin/bash ${RESUME_SCRIPT} ${CHECKPOINT_PATH} 2>&1 | tee -a ${LOG_FILE}'"

sleep 2
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "failed to create tmux session: ${SESSION_NAME}" | tee -a "${LOG_FILE}" >&2
  exit 1
fi

{
  echo "started tmux session: ${SESSION_NAME}"
  echo "view sessions: tmux list-sessions"
  echo "attach: tmux attach -t ${SESSION_NAME}"
  echo "tail log: tail -f ${LOG_FILE}"
} | tee -a "${LOG_FILE}"
