# Evaluation

Evaluation targets the two HIATUS program tasks, scored on the HRS benchmark:

| Task | Level | Question | Primary metric |
|---|---|---|---|
| **TA1** | Document retrieval | Given a query document, rank candidate documents by same-authorship. | Success@k (`S@8`) |
| **TA2** | Author attribution | Given query and candidate author portfolios, score each pair. | Equal Error Rate (`EER`) |

For TA1, lower rank of the true match is better; for TA2, **lower EER is better**.

---

## Restricted data access

HRS (HIATUS Research Set) is **restricted IARPA HIATUS-program data** and is
not distributed with this repository. Evaluation requires access to an HRS
mount. Dataset paths are defined in `authorship/evaluation/constants.py` and
default to the UO cluster layout; override them without editing code:

```bash
export HRS_DATA_ROOT=/path/to/HRS        # HRS2 / HRS3 root (default: /gpfs/home/cuongp/hiatus-libs/HRS)
export HRS1_DATA_ROOT=/path/to/HRS/TA1   # HRS1 TA1 root    (default: /home/hieum/uonlp/avae/data/HRS/TA1)
```

`TA2-performer-config.yaml` (repo root) holds the TA2 decision threshold
(`TA2.Threshold: 0.8`) consumed by `hiatus-metrics`-based TA2 scoring; its
path is exposed as `constants.PERFORMER_CONFIG_PATH`.

### Metrics backend

`authorship/evaluation/evaluator.py` prefers the official **`hiatus-metrics`**
library (`hiatus.featurespace.metrics`, `hiatus.attribution.metrics`) when
importable, and otherwise falls back to equivalent local implementations of
S@k and EER. The fallback is convenient for development but **report official
numbers from the `hiatus-metrics` path** — install the restricted `hiatus`
package (see [requirements.txt](../requirements.txt)) if you are an
authorized program member.

### Genres

Genre keys (`HRS2.1`, `HRS3.1`, `en_cross_genre_short`, `ru_cross_genre_long`,
`zh_cross_genre_medium`, …) are defined in `constants.py`. The genre list is
resolved in this order:

1. `--genres` on the CLI,
2. `evaluation.genres` in `<config_dir>/config.yaml`,
3. a built-in default (HRS3 English + Chinese sets).

---

## Core CLI: `authorship.evaluation.eval_hrs`

Two subcommands:

```bash
# Evaluate one checkpoint
python -m authorship.evaluation.eval_hrs eval   --checkpoint_path <ckpt> ...

# Poll a directory and evaluate each new checkpoint as it appears
python -m authorship.evaluation.eval_hrs watch  --checkpoint_dir <dir> --poll_interval 60 ...
```

Common arguments:

| Arg | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `embedder` or `reranker` |
| `--config_dir` | *(required)* | Directory containing `config.yaml` |
| `--output_dir` | `eval_results` | Where JSON/TXT results are written |
| `--genres` | from config | Space-separated genre keys |
| `--batch_size` | 32 | Encoding batch size |
| `--max_length` | 512 | Token truncation length |

Reranker-mode arguments (`--mode reranker` **requires** the first two):

| Arg | Default | Description |
|---|---|---|
| `--embedder_config_dir` | — | Embedder config dir (required) |
| `--embedder_checkpoint_path` | — | Embedder checkpoint (required) |
| `--top_k` | 16 | Candidates retrieved and passed to the reranker |
| `--reranker_weight` | 0.5 | `score = w * P(yes) + (1-w) * cos_norm` |
| `--reranker_agg` | `max` | TA2 doc-pair → author aggregation: `max`, `mean`, `topk_mean` |
| `--reranker_agg_topk` | 3 | `k` when `--reranker_agg topk_mean` |
| `--no_compare_embedder` | off | Skip the embedder-only baseline (system metrics only) |

W&B arguments: `--wandb_project`, `--wandb_entity`, `--wandb_run_name`,
`--wandb_group`, `--wandb_run_id` (resume an existing run), and
`--wandb_log_to_new_run` (log to a fresh run instead of resuming). Disable
logging entirely with `WANDB_MODE=disabled`.

**Outputs** per checkpoint, keyed on the checkpoint filename stem:

- `<output_dir>/<stem>.json` — raw metrics
- `<output_dir>/<stem>.txt` — "Embedder-only vs Embedder + Reranker" comparison table (reranker mode, unless `--no_compare_embedder`)

`eval` handles `SIGINT`/`SIGTERM` gracefully: the first interrupt finishes the
current genre and then stops; a second exits immediately.

---

## Wrapper scripts

### Single checkpoint — `evaluate_single.sh`

```bash
# embedder
bash scripts/evaluate_single.sh \
  --mode embedder \
  --config_dir outputs/embedder-v0 \
  --checkpoint_path outputs/embedder-v0/checkpoint_step6000.ckpt

# reranker
bash scripts/evaluate_single.sh \
  --mode reranker \
  --config_dir outputs/reranker-v0 \
  --checkpoint_path outputs/reranker-v0/checkpoint_step2000.ckpt \
  --embedder_config_dir outputs/embedder-v0 \
  --embedder_checkpoint_path outputs/embedder-v0/checkpoint_step6000.ckpt \
  --top_k 32 \
  --reranker_weight 0.7
```

W&B logging is opt-in here (only enabled if `--wandb_project` is passed).

### Watch a training run — `evaluate_checkpoints.sh`

Polls `--checkpoint_dir` and evaluates each new checkpoint into a **fresh
W&B eval run** (`--wandb_project` is required).

```bash
bash scripts/evaluate_checkpoints.sh \
  --mode embedder \
  --config_dir outputs/embedder-8B-v2 \
  --checkpoint_dir outputs/embedder-8B-v2 \
  --wandb_project authorship-embedder-eval

bash scripts/evaluate_checkpoints.sh \
  --mode reranker \
  --config_dir outputs/reranker-v0 \
  --checkpoint_dir outputs/reranker-v0 \
  --embedder_config_dir outputs/embedder-v0 \
  --embedder_checkpoint_path outputs/embedder-v0/checkpoint_step6000.ckpt \
  --wandb_project authorship-reranker-eval
```

### All reranker checkpoints, multi-GPU — `evaluate_reranker.sh`

Exports (if needed) and evaluates **every** `checkpoint_step*.ckpt` in a
reranker output dir, one checkpoint per GPU in parallel via a work-stealing
pool. Writes local JSON/TXT only — W&B is force-disabled.

```bash
bash scripts/evaluate_reranker.sh                 # export + eval all
bash scripts/evaluate_reranker.sh --skip-export   # eval only (exports must exist)
bash scripts/evaluate_reranker.sh --export-only   # export only
```

Configuration is entirely via env vars:

| Var | Default | Description |
|---|---|---|
| `EMBEDDER_DIR` | `outputs/merged-8B.v3-s8-wins` | Embedder dir (needs `config.yaml` + sharded safetensors) |
| `RERANKER_DIR` | `outputs/reranker-v1` | Reranker run dir to sweep |
| `TOP_K` | 64 | Retrieval window handed to the reranker |
| `RERANKER_WEIGHT` | 0.25 | Interpolation weight |
| `RERANKER_AGG` / `RERANKER_AGG_TOPK` | `topk_mean` / 16 | TA2 aggregation |
| `GPUS` | auto (`nvidia-smi`) | Explicit GPU pool, e.g. `"0,1,2,3"` |
| `MAX_PARALLEL` | #GPUs | Cap concurrent evals |

Results land in `$RERANKER_DIR/hrs_eval/` (`.json`, `.txt`, per-checkpoint
`.eval.log`, plus an orchestration `eval.log`). Completed checkpoints are
skipped on re-run, so the script is resumable.

### Interpolation-weight sweep — `sweep_reranker_weight.sh`

Sweeps `reranker_weight` on a **single** exported checkpoint to see how much
reranker signal survives interpolation with the embedder score. One weight per
GPU in parallel.

```bash
bash scripts/sweep_reranker_weight.sh
CKPT=outputs/reranker-v9/exported/checkpoint_step21314.pt bash scripts/sweep_reranker_weight.sh
WEIGHTS="0.1 0.3 0.6 1.0" bash scripts/sweep_reranker_weight.sh
```

Each weight gets its own output subdir (`w<W>/`), plus a rolled-up
`sweep_summary.txt` with the MEAN row per weight for quick comparison.

---

## Checkpoint export

Reranker training writes **sharded** Fabric/FSDP checkpoints
(`checkpoint_step{N}.ckpt/` containing distcp shards + `meta.pt`). Evaluation
and inference need a flat, model-only `.pt`. Consolidate on CPU (no GPU
required):

```bash
conda run -n hiatus-phase3 python scripts/export_reranker_fabric_ckpt.py \
  --checkpoint outputs/reranker-v1/checkpoint_step2000.ckpt \
  --output outputs/reranker-v1/exported/checkpoint_step2000.pt
```

`evaluate_reranker.sh` runs this automatically for any checkpoint that has a
`.done` sentinel and no existing export. (The Lightning CLI equivalent,
`fabric consolidate <ckpt>`, writes a `.ckpt.consolidated` from which the
model state dict must then be extracted manually.)

---

## Analysis helpers

### `summarize_hrs_eval.py`

Reprints the embedder vs. embedder+reranker comparison table from saved eval
JSON — useful for reviewing many checkpoints at once without re-running eval.

```bash
python scripts/summarize_hrs_eval.py outputs/reranker-v1/hrs_eval/checkpoint_step8000.json
python scripts/summarize_hrs_eval.py outputs/reranker-v1/hrs_eval/*.json
```

### `diagnose_reranker_pyes.py`

Dumps the reranker's `P(yes)` distribution on one genre, split by label, to
distinguish **reranker transfer failure** from a **downstream wiring bug**:

- Same-author and different-author distributions overlap heavily (low AUC), or
  both saturate near 1.0 → the reranker isn't discriminating on this genre
  (domain-transfer failure).
- They separate cleanly here but the full system still loses to the embedder
  in eval → the problem is downstream (aggregation or score interpolation).

```bash
conda run -n hiatus-phase3 python scripts/diagnose_reranker_pyes.py \
    --reranker_config outputs/reranker-v8/config.yaml \
    --reranker_checkpoint outputs/reranker-v8/checkpoint_step17314.ckpt \
    --genre HRS2.101 --n_pairs 1000
```

No embedder is needed — this scores pairs with the reranker alone.
