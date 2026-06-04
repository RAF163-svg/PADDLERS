#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${GID15_DATASET_ROOT:-}" ]; then
  echo "GID15_DATASET_ROOT is not set" >&2
  exit 2
fi

if [ ! -d "${GID15_DATASET_ROOT}" ]; then
  echo "GID15_DATASET_ROOT does not exist: ${GID15_DATASET_ROOT}" >&2
  exit 2
fi

REQUIRED_FILES=(
  "${PROJECT_ROOT}/paddlers/unet_data_split/train.py"
  "${PROJECT_ROOT}/paddlers/unet_data_split/cluster_routed_dataset.py"
  "${PROJECT_ROOT}/paddlers/unet_data_split/unet_routed_model.py"
  "${PROJECT_ROOT}/release/configs/unet_singlehead_portable.json"
  "${PROJECT_ROOT}/release/configs/unet_multihead_portable.json"
  "${PROJECT_ROOT}/release/data_specs/gid15_clustered_dataset/manifests/sample_cluster_mapping.relative.csv"
  "${PROJECT_ROOT}/release/data_specs/gid15_clustered_dataset/stats/cluster_summary.json"
  "${PROJECT_ROOT}/release/data_specs/gid15_dataset/labels.txt"
)

for item in "${REQUIRED_FILES[@]}"; do
  if [ ! -e "${item}" ]; then
    echo "missing required file: ${item}" >&2
    exit 2
  fi
done

python -m py_compile \
  "${PROJECT_ROOT}/paddlers/unet_data_split/train.py" \
  "${PROJECT_ROOT}/paddlers/unet_data_split/cluster_routed_dataset.py" \
  "${PROJECT_ROOT}/paddlers/unet_data_split/unet_routed_model.py" \
  "${PROJECT_ROOT}/paddlers/unet_data_split/eval_best_model.py" \
  "${PROJECT_ROOT}/paddlers/unet_data_split/infer_best_model.py"

python "${PROJECT_ROOT}/paddlers/unet_data_split/train.py" \
  --config "${PROJECT_ROOT}/release/configs/unet_singlehead_portable.json" \
  --save-dir "${PROJECT_ROOT}/release/validation/singlehead_prepare" \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --max-test-samples 1

python "${PROJECT_ROOT}/paddlers/unet_data_split/train.py" \
  --config "${PROJECT_ROOT}/release/configs/unet_multihead_portable.json" \
  --save-dir "${PROJECT_ROOT}/release/validation/multihead_prepare" \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --max-test-samples 1

echo "release_check=passed"
