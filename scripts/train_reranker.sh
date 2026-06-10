#!/bin/bash
# Train the authorship reranker
#
# Usage: bash scripts/train_reranker.sh [config_overrides...]

set -e

CONFIG_DIR="${CONFIG_DIR:-configs/reranker}"
CONFIG_NAME="${CONFIG_NAME:-default}"

echo "Training reranker with config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"

# Reduce CUDA allocator fragmentation; helps the FSDP checkpoint all-gather
# spike fit within GPU memory at the boundary of OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m authorship.training.train_reranker \
    --config-path "../../${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "$@"
