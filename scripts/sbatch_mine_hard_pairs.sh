#!/bin/bash
# SLURM array job — Dense hard-pair mining for all authorship datasets.
# One array element per dataset; language splits that don't exist are skipped.
#
# Submit:
#   sbatch scripts/sbatch_mine_hard_pairs.sh
#
# Overrides (env vars):
#   EMBEDDER_CONFIG_DIR=outputs/my-model sbatch scripts/sbatch_mine_hard_pairs.sh

#SBATCH --job-name=mine_pairs
#SBATCH --array=0-22%6
#SBATCH --partition=preempt,gpulong,cisds
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu-80gb|gpu-40gb"
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=logs/mine_%A_%a.log
#SBATCH --error=logs/mine_%A_%a.log

set -euo pipefail

mkdir -p logs

DATASETS=(
    # "Hieuman/ru_KP" x
    # "Hieuman/hiatus-imdb" x 
    # "Hieuman/ru_STX" x
    # "Hieuman/MUD" x
    # "Hieuman/movie_reviews" x
    # "Hieuman/blog_authorship" x
    # "Hieuman/stihi_ru" x
    # "Hieuman/goodreads" x
    # "Hieuman/yelp_review" x
    # "Hieuman/proza_ru" x
    # "Hieuman/u-sticker" x
    # "Hieuman/ru_reddit_dump" x
    # "Hieuman/ru_telegram" x
    # "Hieuman/douban_reviews" x
    
    # "Hieuman/STX" x
    # "Hieuman/weibo_full" x
    # "Hieuman/dianping_review" x
    # "Hieuman/wiki_en_small" x
    # "Hieuman/HNI" x
    # "Hieuman/wiki_ru" x
    # "Hieuman/nyt_comments" x
    # "Hieuman/jd_reviews" x
    # "Hieuman/yt_comments" x
    # "Hieuman/reddit_dump" x
    # "Hieuman/exorde" x
    # "Hieuman/amazon_reviews" x
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
NAME="${DATASET#Hieuman/}"
OUTPUT_DIR="${OUT_ROOT:-./data}/${NAME}_hard"
EMBEDDER_CONFIG_DIR="${EMBEDDER_CONFIG_DIR:-outputs/merged-4B.v4-eer-wins}"
EMBEDDING_DIR="${EMBEDDING_DIR:-./data/embeddings}"
NUM_GPUS=4

echo "========================================"
echo "  Task          : ${SLURM_ARRAY_TASK_ID}"
echo "  Dataset       : ${DATASET}"
echo "  Output        : ${OUTPUT_DIR}"
echo "  Embedder      : ${EMBEDDER_CONFIG_DIR}"
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
