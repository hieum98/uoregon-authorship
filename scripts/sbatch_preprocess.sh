#!/bin/bash
# SLURM array job — BM25 hard-negative mining for all authorship datasets.
# One array element per dataset; language splits that don't exist are skipped.
#
# Submit:
#   sbatch scripts/sbatch_preprocess.sh
#
# Override top-k or output root:
#   TOP_K=256 OUT_ROOT=./data/bm25 sbatch scripts/sbatch_preprocess.sh

#SBATCH --job-name=bm25_prep
#SBATCH --array=0-25%10
#SBATCH --partition=computelong
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

set -euo pipefail

mkdir -p logs

DATASETS=(
    "Hieuman/ru_KP"
    "Hieuman/weibo_full"
    "Hieuman/hiatus-imdb"
    "Hieuman/yt_comments"
    "Hieuman/ru_STX"
    "Hieuman/MUD"
    "Hieuman/reddit_dump"
    "Hieuman/dianping_review"
    "Hieuman/wiki_en_small"
    "Hieuman/HNI"
    "Hieuman/wiki_ru"
    "Hieuman/nyt_comments"
    "Hieuman/movie_reviews"
    "Hieuman/jd_reviews"
    "Hieuman/blog_authorship"
    "Hieuman/stihi_ru"
    "Hieuman/goodreads"
    "Hieuman/exorde"
    "Hieuman/douban_reviews"
    "Hieuman/yelp_review"
    "Hieuman/proza_ru"
    "Hieuman/amazon_reviews"
    "Hieuman/u-sticker"
    "Hieuman/ru_reddit_dump"
    "Hieuman/ru_telegram"
    "Hieuman/STX"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
NAME="${DATASET#Hieuman/}"
OUTPUT_DIR="${OUT_ROOT:-./data}/${NAME}"
TOP_K="${TOP_K:-512}"

echo "========================================"
echo "  Task     : ${SLURM_ARRAY_TASK_ID}"
echo "  Dataset  : ${DATASET}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Top-K    : ${TOP_K}"
echo "  Node     : $(hostname)"
echo "========================================"

conda run --no-capture-output -n hiatus-phase3 \
    python -m authorship.preprocessing.bm25_mining \
        --dataset_name  "${DATASET}" \
        --output_dir    "${OUTPUT_DIR}" \
        --languages     en zh ar ru es de fr \
        --top_k         "${TOP_K}"

echo "Done: ${DATASET}"
