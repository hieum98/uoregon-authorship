#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for authorship.tools.model_merge
# Edit paths/weights below or pass fully custom args to python module.

python -m authorship.tools.model_merge \
  --models '["/path/to/model_a.pt", "/path/to/model_b.pt"]' \
  --weights '[0.5, 0.5]' \
  --base-model "/path/to/base_model.pt" \
  --merge-method "ties" \
  --density '[1, 0.7, 0.1]' \
  --config-path "configs/mergekit/model_merging.yaml" \
  --output-dir "outputs/merged-model"

# Add --run-merge to execute mergekit-yaml, otherwise command is printed only.
