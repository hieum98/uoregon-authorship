#!/bin/bash
# SLURM job — Reranker training on n0999 (4x H100-80G)
#
# Submit:
#   sbatch scripts/sbatch_train_reranker.sh
#
# Override config:
#   CONFIG_NAME=local_h100 sbatch scripts/sbatch_train_reranker.sh
#
# Pass extra Hydra overrides after --:
#   sbatch scripts/sbatch_train_reranker.sh -- training.lr=2e-5

#SBATCH --job-name=reranker-v1
#SBATCH --nodelist=n0999
#SBATCH --partition=cisds
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=logs/reranker_%j.log
#SBATCH --error=logs/reranker_%j.log

set -euo pipefail

mkdir -p logs

CONFIG_DIR="${CONFIG_DIR:-configs/reranker}"
CONFIG_NAME="${CONFIG_NAME:-local_h100}"

echo "========================================"
echo "  Job            : ${SLURM_JOB_ID}"
echo "  Node           : $(hostname)"
echo "  GPUs           : ${CUDA_VISIBLE_DEVICES:-all}"
echo "  Config         : ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "========================================"

cd /gpfs/projects/uonlp/hieum/uoregon-authorship

# Reduce CUDA allocator fragmentation; helps the FSDP checkpoint all-gather
# spike fit within GPU memory at the boundary of OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

conda run --no-capture-output -n hiatus-phase3 \
    python -m authorship.training.train_reranker \
        --config-path "../../${CONFIG_DIR}" \
        --config-name "${CONFIG_NAME}" \
        "$@"

echo "Training complete."
