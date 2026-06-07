#!/usr/bin/env bash
# mine_hard_pairs.sh — Dense hard-pair mining for reranker training.
#
# Encodes all documents with the trained embedder, builds a FAISS index, and
# writes two new columns to the dataset:
#   hard_negative_docIDs  — cross-author pairs with highest cosine similarity
#   hard_positive_docIDs  — same-author pairs with lowest cosine similarity
#
# Usage:
#   bash scripts/mine_hard_pairs.sh [DATASET_NAME] [OUTPUT_DIR] [LANGUAGES] \
#       [TOP_K_NEG] [TOP_K_POS] [EMBEDDER_CONFIG_DIR] [EMBEDDING_DIR] [NUM_GPUS] \
#       [NLIST] [NPROBE] [NEG_BUFFER]
#
# All arguments are optional; defaults are shown below. Tunables can also be
# overridden via env vars of the same name.
#
# Use a HuggingFace embedder instead of a trained authorship checkpoint:
#   HF_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B bash scripts/mine_hard_pairs.sh
#
# Examples:
#   # Single GPU, default embedder checkpoint
#   bash scripts/mine_hard_pairs.sh
#
#   # 4 GPUs, custom dataset and output
#   bash scripts/mine_hard_pairs.sh Hieuman/reddit_bm25 ./data/reddit_hard 'en' \
#       512 50 outputs/merged-4B.v4-eer-wins ./data/embeddings 4
#
#   # HF Qwen3-Embedding-0.6B (single GPU)
#   HF_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B \
#       bash scripts/mine_hard_pairs.sh Hieuman/reddit_bm25 ./data/reddit_hard_qwen en
#
#   # Larger over-fetch buffer for prolific-author corpora
#   NEG_BUFFER=16 bash scripts/mine_hard_pairs.sh

set -euo pipefail

DATASET_NAME="${1:-${DATASET_NAME:-Hieuman/reddit_bm25}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-./data/reddit_hard_pairs}}"
LANGUAGES="${3:-${LANGUAGES:-en}}"           # space-separated, e.g. 'en zh ar'
TOP_K_NEG="${4:-${TOP_K_NEG:-512}}"          # hard negatives to store per query
TOP_K_POS="${5:-${TOP_K_POS:-50}}"           # hard positives to store per query
EMBEDDER_CONFIG_DIR="${6:-${EMBEDDER_CONFIG_DIR:-outputs/merged-4B.v4-eer-wins}}"
EMBEDDING_DIR="${7:-${EMBEDDING_DIR:-./data/embeddings}}"
NUM_GPUS="${8:-${NUM_GPUS:-1}}"
NLIST="${9:-${NLIST:-65536}}"                # auto-clamped to N//39 inside the script
NPROBE="${10:-${NPROBE:-128}}"
NEG_BUFFER="${11:-${NEG_BUFFER:-4}}"         # fetch_k = TOP_K_NEG * NEG_BUFFER
HF_EMBEDDER_MODEL="${HF_EMBEDDER_MODEL:-}"
HF_EMBEDDER_INSTRUCT="${HF_EMBEDDER_INSTRUCT:-}"
HF_EMBEDDER_ATTN="${HF_EMBEDDER_ATTN:-}"

EMBEDDER_ARGS=(--embedder-config-dir "${EMBEDDER_CONFIG_DIR}")
if [[ -n "${HF_EMBEDDER_MODEL}" ]]; then
    EMBEDDER_ARGS=(--hf-embedder-model "${HF_EMBEDDER_MODEL}")
    if [[ -n "${HF_EMBEDDER_INSTRUCT}" ]]; then
        EMBEDDER_ARGS+=(--hf-embedder-instruct "${HF_EMBEDDER_INSTRUCT}")
    fi
    if [[ -n "${HF_EMBEDDER_ATTN}" ]]; then
        EMBEDDER_ARGS+=(--hf-embedder-attn "${HF_EMBEDDER_ATTN}")
    fi
fi

echo "=========================================="
echo " Dense Hard-Pair Mining"
echo "=========================================="
echo "  Dataset         : ${DATASET_NAME}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Languages       : ${LANGUAGES}"
echo "  Top-K negatives : ${TOP_K_NEG}"
echo "  Top-K positives : ${TOP_K_POS}"
if [[ -n "${HF_EMBEDDER_MODEL}" ]]; then
echo "  HF embedder     : ${HF_EMBEDDER_MODEL}"
else
echo "  Embedder config : ${EMBEDDER_CONFIG_DIR}"
fi
echo "  Embedding cache : ${EMBEDDING_DIR}"
echo "  GPUs            : ${NUM_GPUS}"
echo "  FAISS nlist     : ${NLIST}"
echo "  FAISS nprobe    : ${NPROBE}"
echo "  Neg buffer      : ${NEG_BUFFER}"
echo "=========================================="

# shellcheck disable=SC2086
PYTHONUNBUFFERED=1 conda run --no-capture-output -n hiatus-phase3 \
    torchrun \
        --nproc-per-node="${NUM_GPUS}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:$((29500 + RANDOM % 1000)) \
        -m authorship.preprocessing.embedder_mining \
            --dataset-name "${DATASET_NAME}" \
            --output-dir "${OUTPUT_DIR}" \
            --languages ${LANGUAGES} \
            "${EMBEDDER_ARGS[@]}" \
            --top-k-neg "${TOP_K_NEG}" \
            --top-k-pos "${TOP_K_POS}" \
            --embedding-dir "${EMBEDDING_DIR}" \
            --nlist "${NLIST}" \
            --nprobe "${NPROBE}" \
            --neg-buffer "${NEG_BUFFER}"
