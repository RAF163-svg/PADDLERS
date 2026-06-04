#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 singlehead|multihead [SAVE_DIR] [extra train args...]" >&2
  exit 2
fi

MODE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

case "${MODE}" in
  singlehead)
    CONFIG_PATH="${PROJECT_ROOT}/release/configs/unet_singlehead_portable.json"
    DEFAULT_SAVE_DIR="${PROJECT_ROOT}/release/runs/unet_singlehead"
    ;;
  multihead)
    CONFIG_PATH="${PROJECT_ROOT}/release/configs/unet_multihead_portable.json"
    DEFAULT_SAVE_DIR="${PROJECT_ROOT}/release/runs/unet_multihead"
    ;;
  *)
    echo "unknown mode: ${MODE}. expected singlehead or multihead" >&2
    exit 2
    ;;
esac

SAVE_DIR="${DEFAULT_SAVE_DIR}"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SAVE_DIR="$1"
  shift
fi

if [ -z "${GID15_DATASET_ROOT:-}" ]; then
  echo "GID15_DATASET_ROOT is not set" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"

export PROJECT_ROOT

bash "${PROJECT_ROOT}/paddlers/unet_data_split/start_train_nohup.sh" \
  "${CONFIG_PATH}" \
  "${SAVE_DIR}" \
  "$@"
