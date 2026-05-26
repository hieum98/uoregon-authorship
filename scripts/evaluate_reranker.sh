#!/usr/bin/env bash
# Evaluate all reranker-v1 checkpoints with merged-8B embedder on HRS (local JSON only, no W&B).
#
# Prereqs:
#   - conda env: hiatus-phase3
#   - HRS data paths in authorship/evaluation/constants.py reachable from this machine
#   - GPUs: export is CPU-only; eval uses 1 GPU by default (set CUDA_VISIBLE_DEVICES)
#
# Usage:
#   cd /home/hieum/uonlp/uoregon-authorship
#   bash scripts/evaluate_reranker_v1_all.sh              # export (if needed) + eval all
#   bash scripts/evaluate_reranker_v1_all.sh --skip-export  # eval only (exported .pt must exist)
#   bash scripts/evaluate_reranker_v1_all.sh --export-only  # export only
#
# Outputs:
#   outputs/reranker-v1/exported/checkpoint_step{N}.pt
#   outputs/reranker-v1/hrs_eval/checkpoint_step{N}.json
#   outputs/reranker-v1/hrs_eval/eval.log

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
TOP_K="${TOP_K:-256}"
RERANKER_WEIGHT="${RERANKER_WEIGHT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
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

eval_one() {
  local pt_path="$1"
  local stem
  stem="$(basename "$pt_path" .pt)"
  local result_json="$EVAL_DIR/${stem}.json"

  if [[ -f "$result_json" ]]; then
    log "Skip eval (exists): $result_json"
    return 0
  fi
  if [[ ! -f "$pt_path" ]]; then
    log "Skip eval (missing export): $pt_path"
    return 1
  fi

  log "Evaluating full system: $stem (see tqdm bars below)"
  echo "========== Eval: $stem =========="
  "${PYTHON[@]}" -m authorship.evaluation.eval_hrs eval \
    --mode reranker \
    --config_dir "$RERANKER_DIR" \
    --checkpoint_path "$pt_path" \
    --embedder_config_dir "$EMBEDDER_DIR" \
    --embedder_checkpoint_path "$EMBEDDER_DIR" \
    --output_dir "$EVAL_DIR" \
    --top_k "$TOP_K" \
    --reranker_weight "$RERANKER_WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    2>&1 | tee -a "$LOG_FILE"
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

for ckpt in "${CKPT_DIRS[@]}"; do
  stem="$(basename "$ckpt")"
  eval_one "$EXPORT_DIR/${stem%.ckpt}.pt" || true
done

log "Done. Results: $EVAL_DIR/*.json"
