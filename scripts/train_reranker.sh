#!/bin/bash
# Train the authorship reranker
#
# Usage: bash scripts/train_reranker.sh [config_overrides...]

set -e

CONFIG_DIR="${CONFIG_DIR:-configs/reranker}"
CONFIG_NAME="${CONFIG_NAME:-default}"

echo "Training reranker with config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"

python -m authorship.training.train_reranker \
    --config-path "../../${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "$@"
