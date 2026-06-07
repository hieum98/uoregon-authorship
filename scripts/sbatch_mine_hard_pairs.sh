#!/bin/bash
# SLURM array job — Dense hard-pair mining for all authorship datasets.
# One array element per dataset; language splits that don't exist are skipped.
#
# Submit:
#   sbatch scripts/sbatch_mine_hard_pairs.sh
#
# Overrides (env vars):
#   EMBEDDER_CONFIG_DIR=outputs/my-model sbatch scripts/sbatch_mine_hard_pairs.sh
#   HF_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B sbatch scripts/sbatch_mine_hard_pairs.sh

#SBATCH --job-name=mine_pairs
#SBATCH --array=0-22%6
#SBATCH --partition=preempt,gpulong,cisds
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu-80gb"
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=logs/mine_%A_%a.log
#SBATCH --error=logs/mine_%A_%a.log

set -euo pipefail

mkdir -p logs

DATASETS=(
    # "Hieuman/ru_KP" 
    # "Hieuman/hiatus-imdb" 
    # "Hieuman/ru_STX" 
    # "Hieuman/movie_reviews" 
    # "Hieuman/blog_authorship" 
    # "Hieuman/stihi_ru" 
    # "Hieuman/goodreads" 
    # "Hieuman/yelp_review" 
    # "Hieuman/proza_ru" 
    # "Hieuman/u-sticker" 
    # "Hieuman/ru_reddit_dump" 
    # "Hieuman/ru_telegram" 
    # "Hieuman/douban_reviews" 
    
    "Hieuman/MUD" 
    "Hieuman/STX" 
    "Hieuman/weibo_full" 
    "Hieuman/dianping_review" 
    "Hieuman/wiki_en_small" 
    "Hieuman/HNI" 
    "Hieuman/wiki_ru" 
    "Hieuman/nyt_comments" 
    "Hieuman/jd_reviews" 
    "Hieuman/yt_comments" 
    "Hieuman/reddit_dump" 
    "Hieuman/exorde" 
    "Hieuman/amazon_reviews" 
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
NAME="${DATASET#Hieuman/}"
OUTPUT_DIR="${OUT_ROOT:-./data-topics}/${NAME}_hard"
EMBEDDER_CONFIG_DIR="${EMBEDDER_CONFIG_DIR:-outputs/merged-4B.v4-eer-wins}"
HF_EMBEDDER_MODEL="${HF_EMBEDDER_MODEL:-}"
EMBEDDING_DIR="${EMBEDDING_DIR:-./data-topics/embeddings}"
NUM_GPUS=4

echo "========================================"
echo "  Task          : ${SLURM_ARRAY_TASK_ID}"
echo "  Dataset       : ${DATASET}"
echo "  Output        : ${OUTPUT_DIR}"
if [[ -n "${HF_EMBEDDER_MODEL}" ]]; then
echo "  HF embedder   : ${HF_EMBEDDER_MODEL}"
else
echo "  Embedder      : ${EMBEDDER_CONFIG_DIR}"
fi
echo "  Embedding dir : ${EMBEDDING_DIR}"
echo "  Node          : $(hostname)"
echo "  GPUs          : $(echo $CUDA_VISIBLE_DEVICES)"
echo "========================================"

bash scripts/mine_hard_pairs.sh \
    "${DATASET}" \
    "${OUTPUT_DIR}" \
    "en zh ar ru es de fr" \
    512 \
    50 \
    "${EMBEDDER_CONFIG_DIR}" \
    "${EMBEDDING_DIR}" \
    "${NUM_GPUS}"

echo "Done: ${DATASET}"
