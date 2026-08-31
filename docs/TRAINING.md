# Training

Two independently-trained models make up the pipeline: an **embedder**
(retriever/bi-encoder) and a **reranker** (cross-encoder). See
[PREPROCESSING.md](PREPROCESSING.md) for how their input datasets are built,
and the [full pipeline diagram](../README.md#full-pipeline) for the stage
ordering.

All training is [Hydra](https://hydra.cc/)-driven — every config key can be
overridden from the CLI without editing YAML.

```bash
conda run -n hiatus-phase3 bash scripts/train_embedder.sh \
  training.devices=1 \
  training.epochs=1 \
  data.num_train_examples=5000
```

Config resolution: `${CONFIG_DIR:-configs/embedder}/${CONFIG_NAME:-default}.yaml`
(reranker: `configs/reranker`). Use a different base config with
`CONFIG_NAME=local_h100 bash scripts/train_reranker.sh` (an alternate config
already exists at `configs/reranker/local_h100.yaml`).

Checkpoints are written to `training.output_dir` (default `outputs/<run_name>`)
and logs to `wandb/` (project set by `training.logger.project`).

---

## 1. Train Embedder

`authorship/training/train_embedder.py`, wrapped by `scripts/train_embedder.sh`.

```bash
conda run -n hiatus-phase3 bash scripts/train_embedder.sh [hydra_overrides...]

# single GPU
conda run -n hiatus-phase3 bash scripts/train_embedder.sh training.devices=1
```

Trains a bi-encoder retriever with contrastive loss (`InfoNCE`-style, see
`authorship/training/losses.py`) using BM25-mined (or dense-mined) hard
negatives. Consumes `sameAuthor_docIDs` / `hard_negative_docIDs` columns
per [PREPROCESSING.md](PREPROCESSING.md#1-bm25-hard-negative-mining).

Key config knobs (`configs/embedder/default.yaml`):

| Key | Default | Description |
|---|---|---|
| `model.model_name_or_path` | `Qwen/Qwen3-8B-Base` | Base model |
| `model.is_bidirectional` | `true` | Bidirectional attention for embedding (see `authorship/models/bidirectional.py`) |
| `model.lora.use_lora` | `false` | LoRA adapter training vs. full fine-tune |
| `model.quantization` | `false` | 4-bit quantized base weights (requires `bitsandbytes`, see [README](../README.md#requirements)) |
| `training.devices` | 4 | Number of GPUs |
| `training.epochs` | 2 | Training epochs |
| `training.lr` | 6e-6 | Learning rate |
| `training.precision` | `bf16-true` | Lightning precision mode |
| `training.gc_chunk_size` | 16 | GradCache micro-batch chunk size (see `authorship/training/gradcache.py`) — trades speed for memory when the full contrastive batch doesn't fit |
| `training.checkpoint_interval` | 1000 | Steps between checkpoints |
| `training.resume_checkpoint` | `outputs/merged-8B.v3-eer-wins` | Starting checkpoint (set `training.only_reload_model=true` to load weights only, not optimizer/scheduler state) |
| `data.num_hard_negatives` | 126 | Hard negatives per sample |
| `data.num_train_examples` | 10000 | Cap per dataset (each dataset in `data.dataset_names` is capped independently) |
| `data.max_seq_length` | 256 | Token truncation length |
| `loss.temperature` | 0.05 | Contrastive loss temperature |
| `loss.use_miner` | `true` | Online hard-negative mining within-batch (`pytorch-metric-learning`) |
| `evaluation.genres` | (HRS genre list) | Genres evaluated during training-time validation — requires HRS data access, see [EVALUATION.md](EVALUATION.md#restricted-data-access) |

`data.dataset_names` in `configs/embedder/default.yaml` lists ~26 private
HuggingFace datasets (`Hieuman/*`) used for the reference run — swap in your
own preprocessed dataset(s) for a different training set.

### SLURM

There's no dedicated `sbatch_train_embedder.sh` in this repo; adapt
`scripts/sbatch_train_reranker.sh` (below) — swap the module path to
`authorship.training.train_embedder` and the config dir to `configs/embedder`.

---

## 2. Train Reranker

`authorship/training/train_reranker.py`, wrapped by `scripts/train_reranker.sh`.

```bash
conda run -n hiatus-phase3 bash scripts/train_reranker.sh [hydra_overrides...]

# single GPU
conda run -n hiatus-phase3 bash scripts/train_reranker.sh training.devices=1

# point at the dense-mined dataset from PREPROCESSING.md stage 2
conda run -n hiatus-phase3 bash scripts/train_reranker.sh \
    "data.dataset_names=[./data/reddit_hard_pairs]" \
    training.devices=1
```

Trains a cross-encoder reranker on hard pairs. Sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default (reduces CUDA
allocator fragmentation during the FSDP checkpoint all-gather) — override by
exporting your own value before calling the script.

**How dataset columns are used** (loader prefers dense-mined columns, falls
back automatically so BM25-only datasets remain compatible):

| Column | Training source | Fallback |
|---|---|---|
| `hard_positive_docIDs` | Same-author pairs, lowest cosine similarity first (dense-mined) | `sameAuthor_docIDs` (random) |
| `hard_negative_docIDs` | Cross-author pairs, highest cosine similarity first (dense-mined) | `BM25_retrieved_docIDs` |

Key config knobs (`configs/reranker/default.yaml`):

| Key | Default | Description |
|---|---|---|
| `model.model_name_or_path` | `Qwen/Qwen3-4B-Base` | Base model |
| `model.lora.use_lora` | `false` | LoRA vs. full fine-tune |
| `training.devices` | 4 | Number of GPUs |
| `training.epochs` | 5 | Training epochs |
| `training.lr` | 1.0e-5 | Learning rate |
| `training.gradient_accumulation_steps` | 8 | Grad accumulation steps |
| `training.checkpoint_interval` | 2000 | Steps between checkpoints |
| `data.num_pos_per_query` | 1 | Hard positives per query |
| `data.num_neg_per_query` | 1 | Hard negatives per query |
| `data.num_train_examples` | 50000 | Cap per dataset |
| `data.global_batch_size` | 64 | Global batch size across all GPUs |
| `data.max_seq_length` | 1024 | Token truncation length |
| `evaluation.genres` | (HRS genre list) | Genres evaluated during training-time validation |
| `evaluation.reranker_weight` | 0.5 | Score-interpolation weight used for eval-time reranking, see [INFERENCE.md](INFERENCE.md) |

Reranker checkpoints are saved as sharded Lightning Fabric/FSDP checkpoints
(`checkpoint_step{N}.ckpt/`) — consolidate them to a flat `.pt` before
inference/eval with `scripts/export_reranker_fabric_ckpt.py`
(see [EVALUATION.md](EVALUATION.md#checkpoint-export)).

### SLURM

`scripts/sbatch_train_reranker.sh` — single-node, 4x H100-80G job pinned to
node `n0999`, partition `cisds`.

```bash
sbatch scripts/sbatch_train_reranker.sh
CONFIG_NAME=local_h100 sbatch scripts/sbatch_train_reranker.sh
sbatch scripts/sbatch_train_reranker.sh -- training.lr=2e-5   # extra hydra overrides
```

**Submit from the repository root** — SLURM copies the batch script to a spool
directory, so the script resolves the repo via `$SLURM_SUBMIT_DIR`. Override
explicitly if you submit from elsewhere:

```bash
REPO_ROOT=/path/to/uoregon-authorship sbatch scripts/sbatch_train_reranker.sh
```

Note that `configs/reranker/local_h100.yaml` (this job's default config) lists
`data.dataset_names` as **absolute paths** to a specific machine's mined
datasets. Point them at your own dense-mined output dirs before using it, or
override on the command line.

---

## Config Files Reference

| File | Purpose |
|---|---|
| `configs/embedder/default.yaml` | Reference embedder training config (8x GPU, 26-dataset mixture) |
| `configs/reranker/default.yaml` | Reference reranker training config (4x GPU) |
| `configs/reranker/local_h100.yaml` | Alternate reranker config for single-node H100 runs |
| `configs/mergekit/model_merging.template.yaml` | Template consumed by `authorship/tools/model_merge.py`, see [MODEL_MERGING.md](MODEL_MERGING.md) |
