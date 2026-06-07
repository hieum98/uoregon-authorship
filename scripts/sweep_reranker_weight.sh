#!/usr/bin/env bash
# Sweep reranker_weight on a SINGLE exported checkpoint to diagnose how much
# of the reranker signal is being washed out by the embedder during score
# interpolation (sys_score = w * P(yes) + (1-w) * cos_norm).
#
# Each weight gets its own output subdir so the per-checkpoint JSON/TXT files
# don't collide. The eval CLI keys its output filename on the checkpoint stem,
# so per-weight output_dir is the cleanest separation.
#
# Usage:
#   bash scripts/sweep_reranker_weight.sh                          # defaults: v10/step18000, weights={0.25,0.5,0.75,1.0}
#   CKPT=outputs/reranker-v9/exported/checkpoint_step21314.pt \
#     bash scripts/sweep_reranker_weight.sh
#   WEIGHTS="0.1 0.3 0.6 1.0" bash scripts/sweep_reranker_weight.sh
#
# Multi-GPU: one weight per GPU in parallel (work-stealing pool, same pattern
# as evaluate_reranker.sh).
#
# Outputs (per weight w):
#   <SWEEP_DIR>/w<W>/<stem>.json
#   <SWEEP_DIR>/w<W>/<stem>.txt
#   <SWEEP_DIR>/w<W>/<stem>.eval.log
#   <SWEEP_DIR>/sweep.log                # orchestration log
#   <SWEEP_DIR>/sweep_summary.txt        # MEAN row per weight

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EMBEDDER_DIR="${EMBEDDER_DIR:-outputs/merged-8B.v3-s8-wins}"
RERANKER_DIR="${RERANKER_DIR:-outputs/reranker-v10}"
CKPT="${CKPT:-$RERANKER_DIR/exported/checkpoint_step18000.pt}"
WEIGHTS="${WEIGHTS:-0.25 0.5 0.75 1.0}"

CKPT_STEM="$(basename "$CKPT" .pt)"
SWEEP_DIR="${SWEEP_DIR:-$RERANKER_DIR/hrs_eval_weight_sweep/${CKPT_STEM}}"
LOG_FILE="$SWEEP_DIR/sweep.log"

TOP_K="${TOP_K:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RERANKER_AGG="${RERANKER_AGG:-topk_mean}"
RERANKER_AGG_TOPK="${RERANKER_AGG_TOPK:-16}"

mkdir -p "$SWEEP_DIR"

# Avoid accidental W&B logging — these are diagnostic runs.
export WANDB_MODE=disabled
export WANDB_DISABLED=true

if [[ "${CONDA_DEFAULT_ENV:-}" == "hiatus-phase3" ]]; then
  PYTHON=(python)
elif [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON=("$PYTHON_CMD")
else
  PYTHON=(conda run -n hiatus-phase3 --no-capture-output python)
fi

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

[[ -f "$CKPT" ]] || { echo "Error: checkpoint not found: $CKPT"; exit 1; }
[[ -f "$EMBEDDER_DIR/config.yaml" ]] || { echo "Error: embedder config not found: $EMBEDDER_DIR/config.yaml"; exit 1; }
[[ -f "$EMBEDDER_DIR/model.safetensors.index.json" ]] || { echo "Error: embedder weights not found under $EMBEDDER_DIR"; exit 1; }
[[ -f "$RERANKER_DIR/config.yaml" ]] || { echo "Error: reranker config not found: $RERANKER_DIR/config.yaml"; exit 1; }

read -ra WEIGHT_LIST <<< "$WEIGHTS"

log "Embedder:   $EMBEDDER_DIR"
log "Reranker:   $RERANKER_DIR"
log "Checkpoint: $CKPT"
log "Weights:    ${WEIGHT_LIST[*]}"
log "Sweep dir:  $SWEEP_DIR"

detect_gpus() {
  GPU_LIST=()
  if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
  fi
  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    GPU_LIST=("")
  fi
  if [[ -n "${MAX_PARALLEL:-}" && "${MAX_PARALLEL}" -gt 0 && "${MAX_PARALLEL}" -lt "${#GPU_LIST[@]}" ]]; then
    GPU_LIST=("${GPU_LIST[@]:0:$MAX_PARALLEL}")
  fi
  # Don't allocate more GPU slots than we have weights to evaluate.
  if [[ ${#GPU_LIST[@]} -gt ${#WEIGHT_LIST[@]} ]]; then
    GPU_LIST=("${GPU_LIST[@]:0:${#WEIGHT_LIST[@]}}")
  fi
}

eval_one() {
  local w="$1"
  local gpu="${2:-}"
  local out_dir="$SWEEP_DIR/w${w}"
  local result_json="$out_dir/${CKPT_STEM}.json"
  local ckpt_log="$out_dir/${CKPT_STEM}.eval.log"

  mkdir -p "$out_dir"

  if [[ -f "$result_json" ]]; then
    log "Skip eval w=$w (exists): $result_json"
    return 0
  fi

  local gpu_msg="default GPU"
  [[ -n "$gpu" ]] && gpu_msg="GPU $gpu"
  log "Evaluating w=$w on $gpu_msg (log: $ckpt_log)"

  local -a cmd=(
    "${PYTHON[@]}" -m authorship.evaluation.eval_hrs eval
    --mode reranker
    --config_dir "$RERANKER_DIR"
    --checkpoint_path "$CKPT"
    --embedder_config_dir "$EMBEDDER_DIR"
    --embedder_checkpoint_path "$EMBEDDER_DIR"
    --output_dir "$out_dir"
    --top_k "$TOP_K"
    --reranker_weight "$w"
    --reranker_agg "$RERANKER_AGG"
    --reranker_agg_topk "$RERANKER_AGG_TOPK"
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
  )

  local rc=0
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$ckpt_log" 2>&1 || rc=$?
  else
    "${cmd[@]}" > "$ckpt_log" 2>&1 || rc=$?
  fi

  if [[ $rc -ne 0 ]]; then
    log "ERROR: eval failed for w=$w (rc=$rc); see $ckpt_log"
  else
    log "Done eval: w=$w (table: $out_dir/${CKPT_STEM}.txt)"
  fi
  return $rc
}

run_all() {
  detect_gpus
  local n=${#GPU_LIST[@]}
  log "Evaluating across ${n} GPU slot(s): [${GPU_LIST[*]}]"

  if [[ $n -le 1 ]]; then
    local gpu="${GPU_LIST[0]:-}"
    for w in "${WEIGHT_LIST[@]}"; do
      eval_one "$w" "$gpu" || true
    done
    return
  fi

  local fifo
  fifo="$(mktemp -u)"
  mkfifo "$fifo"
  exec 9<>"$fifo"
  rm -f "$fifo"

  local g
  for g in "${GPU_LIST[@]}"; do
    printf '%s\n' "$g" >&9
  done

  for w in "${WEIGHT_LIST[@]}"; do
    local gpu
    read -r gpu <&9
    {
      eval_one "$w" "$gpu" || true
      printf '%s\n' "$gpu" >&9
    } &
  done
  wait
  exec 9>&-
}

run_all

# Roll up MEAN row per weight into a single comparison file.
SUMMARY="$SWEEP_DIR/sweep_summary.txt"
{
  echo "Checkpoint: $CKPT"
  echo "Embedder:   $EMBEDDER_DIR"
  echo "Weights:    ${WEIGHT_LIST[*]}"
  echo ""
  printf "%-8s %-s\n" "weight" "MEAN row (Emb S@8 | Sys S@8 | Δ S@8 | Emb EER | Sys EER | Δ EER)"
  echo "--------------------------------------------------------------------------------"
  for w in "${WEIGHT_LIST[@]}"; do
    txt="$SWEEP_DIR/w${w}/${CKPT_STEM}.txt"
    if [[ -f "$txt" ]]; then
      mean=$(grep "^MEAN" "$txt" | head -1 | sed 's/^MEAN[[:space:]]*//')
      printf "w=%-6s %s\n" "$w" "$mean"
    else
      printf "w=%-6s (missing: %s)\n" "$w" "$txt"
    fi
  done
} | tee "$SUMMARY"

log "Done. Sweep summary: $SUMMARY"
