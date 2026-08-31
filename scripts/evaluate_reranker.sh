#!/usr/bin/env bash
# Evaluate all reranker-v1 checkpoints with merged-8B embedder on HRS (local JSON only, no W&B).
#
# Prereqs:
#   - conda env: hiatus-phase3
#   - HRS data paths in authorship/evaluation/constants.py reachable from this machine
#   - GPUs: each eval uses 1 GPU. On a multi-GPU box (e.g. n0998) this script
#     runs one checkpoint per GPU in parallel (work-stealing pool).
#
# Usage:
#   bash scripts/evaluate_reranker.sh                 # export (if needed) + eval all (parallel over all GPUs)
#   bash scripts/evaluate_reranker.sh --skip-export   # eval only (exported .pt must exist)
#   bash scripts/evaluate_reranker.sh --export-only   # export only
#
# GPU control (env vars):
#   GPUS="0,1,2,3"        # explicit GPU pool to spread evals across
#   MAX_PARALLEL=2        # cap concurrent evals below the number of GPUs
#   CUDA_VISIBLE_DEVICES  # if set (and GPUS unset), used as the GPU pool
#   (default: auto-detect all GPUs via nvidia-smi; one eval per GPU)
#
# Outputs (per checkpoint):
#   outputs/reranker-v1/exported/checkpoint_step{N}.pt
#   outputs/reranker-v1/hrs_eval/checkpoint_step{N}.json    # raw metrics
#   outputs/reranker-v1/hrs_eval/checkpoint_step{N}.txt     # "Embedder-only vs Embedder + Reranker" table
#   outputs/reranker-v1/hrs_eval/checkpoint_step{N}.eval.log # full per-checkpoint stdout/stderr
#   outputs/reranker-v1/hrs_eval/eval.log                    # orchestration log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EMBEDDER_DIR="${EMBEDDER_DIR:-outputs/merged-8B.v3-s8-wins}"
RERANKER_DIR="${RERANKER_DIR:-outputs/reranker-v1}"
RERANKER_CONFIG="${RERANKER_CONFIG:-$RERANKER_DIR/config.yaml}"
EXPORT_DIR="${EXPORT_DIR:-$RERANKER_DIR/exported}"
EVAL_DIR="${EVAL_DIR:-$RERANKER_DIR/hrs_eval}"
LOG_FILE="${LOG_FILE:-$EVAL_DIR/eval.log}"

# top_k=256 widens the reranker window so retriever recall@top_k → ~100%,
# which keeps the shift+normalize fix from creating an EER floor.
TOP_K="${TOP_K:-64}"
RERANKER_WEIGHT="${RERANKER_WEIGHT:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
# TA2 doc-pair -> author aggregation: max | mean | topk_mean
RERANKER_AGG="${RERANKER_AGG:-topk_mean}"
RERANKER_AGG_TOPK="${RERANKER_AGG_TOPK:-16}"
# Deprecated: export no longer uses multi-GPU Fabric.launch
EXPORT_DEVICES="${EXPORT_DEVICES:-}"

SKIP_EXPORT=0
EXPORT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-export) SKIP_EXPORT=1 ;;
    --export-only) EXPORT_ONLY=1 ;;
  esac
done

mkdir -p "$EXPORT_DIR" "$EVAL_DIR"

# Avoid accidental W&B logging
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# Use activated env when possible (matches train_reranker.sh); else conda run.
if [[ "${CONDA_DEFAULT_ENV:-}" == "hiatus-phase3" ]]; then
  PYTHON=(python)
  PIP=(pip)
elif [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON=("$PYTHON_CMD")
  PIP=(conda run -n hiatus-phase3 pip)
else
  PYTHON=(conda run -n hiatus-phase3 --no-capture-output python)
  PIP=(conda run -n hiatus-phase3 pip)
fi

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

# Resolve the GPU pool to spread evals across.
#   GPUS env > CUDA_VISIBLE_DEVICES env > nvidia-smi auto-detect > single (empty) slot.
# Result is stored in the GPU_LIST array.
detect_gpus() {
  if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
  fi
  # Fallback: a single slot with no explicit pinning.
  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    GPU_LIST=("")
  fi
  # Optionally cap concurrency below the number of available GPUs.
  if [[ -n "${MAX_PARALLEL:-}" && "${MAX_PARALLEL}" -gt 0 && "${MAX_PARALLEL}" -lt "${#GPU_LIST[@]}" ]]; then
    GPU_LIST=("${GPU_LIST[@]:0:$MAX_PARALLEL}")
  fi
}

if [[ ! -f "$EMBEDDER_DIR/config.yaml" ]]; then
  echo "Error: embedder config not found: $EMBEDDER_DIR/config.yaml"
  exit 1
fi
if [[ ! -f "$EMBEDDER_DIR/model.safetensors.index.json" ]]; then
  echo "Error: embedder weights not found under $EMBEDDER_DIR (expected sharded safetensors)"
  exit 1
fi
if [[ ! -f "$RERANKER_CONFIG" ]]; then
  echo "Error: reranker config not found: $RERANKER_CONFIG"
  exit 1
fi

shopt -s nullglob
CKPT_DIRS=("$RERANKER_DIR"/checkpoint_step*.ckpt)
if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "Error: no checkpoint_step*.ckpt directories in $RERANKER_DIR"
  exit 1
fi

log "Embedder: $EMBEDDER_DIR"
log "Reranker checkpoints: ${#CKPT_DIRS[@]} under $RERANKER_DIR"
log "Export dir: $EXPORT_DIR | Eval dir: $EVAL_DIR"

export_one() {
  local fabric_ckpt="$1"
  local stem
  stem="$(basename "$fabric_ckpt")"
  local out_pt="$EXPORT_DIR/${stem%.ckpt}.pt"

  if [[ -f "$out_pt" ]]; then
    log "Skip export (exists): $out_pt"
    return 0
  fi
  if [[ ! -f "${fabric_ckpt}.done" ]]; then
    log "Skip export (no .done sentinel): $fabric_ckpt"
    return 1
  fi

  log "Exporting $fabric_ckpt -> $out_pt (CPU consolidate)"
  if ! "${PYTHON[@]}" scripts/export_reranker_fabric_ckpt.py \
    --checkpoint "$fabric_ckpt" \
    --output "$out_pt" \
    2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: export failed for $fabric_ckpt"
    return 1
  fi
}

# Evaluate one exported checkpoint. Pins to $2 (a GPU id) when provided and
# redirects all eval output to a per-checkpoint log to keep parallel runs readable.
eval_one() {
  local pt_path="$1"
  local gpu="${2:-}"
  local stem
  stem="$(basename "$pt_path" .pt)"
  local result_json="$EVAL_DIR/${stem}.json"
  local ckpt_log="$EVAL_DIR/${stem}.eval.log"

  if [[ -f "$result_json" ]]; then
    log "Skip eval (exists): $result_json"
    return 0
  fi
  if [[ ! -f "$pt_path" ]]; then
    log "Skip eval (missing export): $pt_path"
    return 1
  fi

  local gpu_msg="default GPU"
  [[ -n "$gpu" ]] && gpu_msg="GPU $gpu"
  log "Evaluating $stem on $gpu_msg (log: $ckpt_log)"

  local -a cmd=(
    "${PYTHON[@]}" -m authorship.evaluation.eval_hrs eval
    --mode reranker
    --config_dir "$RERANKER_DIR"
    --checkpoint_path "$pt_path"
    --embedder_config_dir "$EMBEDDER_DIR"
    --embedder_checkpoint_path "$EMBEDDER_DIR"
    --output_dir "$EVAL_DIR"
    --top_k "$TOP_K"
    --reranker_weight "$RERANKER_WEIGHT"
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
    log "ERROR: eval failed for $stem (rc=$rc); see $ckpt_log"
  else
    log "Done eval: $stem (table: $EVAL_DIR/${stem}.txt)"
  fi
  return $rc
}

# Run all evals, spreading them across GPU_LIST. Uses a FIFO as a token
# semaphore: each GPU id is a token; a checkpoint blocks until a token is
# free, runs on that GPU in the background, then returns the token. This is
# work-stealing — a freed GPU immediately picks up the next pending checkpoint.
run_all_evals() {
  detect_gpus
  local n=${#GPU_LIST[@]}
  log "Evaluating across ${n} GPU slot(s): [${GPU_LIST[*]}]"

  if [[ $n -le 1 ]]; then
    local gpu="${GPU_LIST[0]:-}"
    for ckpt in "${CKPT_DIRS[@]}"; do
      local stem; stem="$(basename "$ckpt")"
      eval_one "$EXPORT_DIR/${stem%.ckpt}.pt" "$gpu" || true
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

  for ckpt in "${CKPT_DIRS[@]}"; do
    local gpu
    read -r gpu <&9          # blocks until a GPU token is available
    {
      local stem; stem="$(basename "$ckpt")"
      eval_one "$EXPORT_DIR/${stem%.ckpt}.pt" "$gpu" || true
      printf '%s\n' "$gpu" >&9   # return the token
    } &
  done
  wait
  exec 9>&-
}

EXPORT_FAILED=0
if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  if ! "${PYTHON[@]}" -c "import lightning" 2>/dev/null; then
    echo "Error: lightning required for export. Run: pip install lightning"
    exit 1
  fi
  for ckpt in "${CKPT_DIRS[@]}"; do
    export_one "$ckpt" || EXPORT_FAILED=1
  done
  if [[ "$EXPORT_FAILED" -ne 0 ]]; then
    log "One or more exports failed; see $LOG_FILE"
    exit 1
  fi
fi

if [[ "$EXPORT_ONLY" -eq 1 ]]; then
  log "Export-only finished."
  exit 0
fi

run_all_evals

log "Done. Results: $EVAL_DIR/*.json | Tables: $EVAL_DIR/*.txt"
