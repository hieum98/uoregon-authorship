#!/bin/bash
# Evaluate checkpoints in a directory and log metrics to a fresh W&B eval run.
#
# Usage (embedder):
#   bash scripts/evaluate_checkpoints.sh \
#     --mode embedder \
#     --config_dir outputs/embedder-8B-v2 \
#     --checkpoint_dir outputs/embedder-8B-v2 \
#     --wandb_project authorship-embedder-eval
#
# Usage (reranker):
#   bash scripts/evaluate_checkpoints.sh \
#     --mode reranker \
#     --config_dir outputs/reranker-v0 \
#     --checkpoint_dir outputs/reranker-v0 \
#     --embedder_config_dir outputs/embedder-v0 \
#     --embedder_checkpoint_path outputs/embedder-v0/checkpoint_step6000.ckpt \
#     --wandb_project authorship-reranker-eval

set -euo pipefail

MODE=""
CONFIG_DIR=""
CHECKPOINT_DIR=""
OUTPUT_DIR="eval_results"
POLL_INTERVAL="60"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_GROUP=""
GENRES=()
EXTRA_ARGS=()

EMBEDDER_CONFIG_DIR=""
EMBEDDER_CHECKPOINT_PATH=""
TOP_K="16"
RERANKER_WEIGHT="0.5"
BATCH_SIZE="32"
MAX_LENGTH="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config_dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --poll_interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --wandb_group)
            WANDB_GROUP="$2"
            shift 2
            ;;
        --genres)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GENRES+=("$1")
                shift
            done
            ;;
        --embedder_config_dir)
            EMBEDDER_CONFIG_DIR="$2"
            shift 2
            ;;
        --embedder_checkpoint_path)
            EMBEDDER_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --reranker_weight)
            RERANKER_WEIGHT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODE" || -z "$CONFIG_DIR" || -z "$CHECKPOINT_DIR" ]]; then
    echo "Error: --mode, --config_dir, and --checkpoint_dir are required."
    exit 1
fi

if [[ "$MODE" != "embedder" && "$MODE" != "reranker" ]]; then
    echo "Error: --mode must be either 'embedder' or 'reranker'."
    exit 1
fi

if [[ -z "$WANDB_PROJECT" ]]; then
    echo "Error: --wandb_project is required so eval logs are tracked in a dedicated W&B project/run."
    exit 1
fi

if [[ "$MODE" == "reranker" ]]; then
    if [[ -z "$EMBEDDER_CONFIG_DIR" || -z "$EMBEDDER_CHECKPOINT_PATH" ]]; then
        echo "Error: reranker mode requires --embedder_config_dir and --embedder_checkpoint_path."
        exit 1
    fi
fi

CMD=(
    python -m authorship.evaluation.eval_hrs watch
    --mode "$MODE"
    --config_dir "$CONFIG_DIR"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --output_dir "$OUTPUT_DIR"
    --poll_interval "$POLL_INTERVAL"
    --wandb_project "$WANDB_PROJECT"
    --wandb_log_to_new_run
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
)

if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
    CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
if [[ -n "$WANDB_GROUP" ]]; then
    CMD+=(--wandb_group "$WANDB_GROUP")
fi
if [[ ${#GENRES[@]} -gt 0 ]]; then
    CMD+=(--genres "${GENRES[@]}")
fi

if [[ "$MODE" == "reranker" ]]; then
    CMD+=(
        --embedder_config_dir "$EMBEDDER_CONFIG_DIR"
        --embedder_checkpoint_path "$EMBEDDER_CHECKPOINT_PATH"
        --top_k "$TOP_K"
        --reranker_weight "$RERANKER_WEIGHT"
    )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running checkpoint evaluation watcher with new W&B run logging..."
printf 'Command: %q ' "${CMD[@]}"
echo

"${CMD[@]}"
