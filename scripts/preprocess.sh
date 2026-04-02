#!/bin/bash
# BM25 hard negative mining for authorship datasets
#
# Usage: bash scripts/preprocess.sh

set -e

DATASET_NAME="${1:-Hieuman/reddit_bm25}"
OUTPUT_DIR="${2:-./data/reddit_bm25}"
LANGUAGES="${3:-en}"
TOP_K="${4:-512}"

echo "Mining BM25 hard negatives for ${DATASET_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Languages: ${LANGUAGES}"
echo "  Top-k: ${TOP_K}"

python -m authorship.preprocessing.bm25_mining \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --languages ${LANGUAGES} \
    --top_k ${TOP_K}

echo "Done!"
