#!/bin/bash
# Train the authorship embedder (retriever)
#
# Usage: bash scripts/train_embedder.sh [config_overrides...]

set -e

CONFIG_DIR="${CONFIG_DIR:-configs/embedder}"
CONFIG_NAME="${CONFIG_NAME:-default}"

echo "Training embedder with config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"

python -m authorship.training.train_embedder \
    --config-path "../../${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "$@"
