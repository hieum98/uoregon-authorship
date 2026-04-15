# uoregon-authorship

Authorship attribution and retrieval with a retriever-reranker pipeline.

This repository provides:

- A unified inference API through `AuthorshipModel`.
- Training pipelines for an embedder (retriever) and reranker.
- Preprocessing: BM25 hard-negative mining and dense hard-pair mining.
- Merge utilities for raw PyTorch checkpoints via `mergekit-pytorch`.
- Evaluation utilities for TA1/TA2-style metrics.

## Requirements

- Python 3.10+
- Linux recommended
- GPU strongly recommended for training/merging

Core dependencies are listed in `requirements.txt` and include:

- `torch`, `transformers`, `lightning`, `hydra-core`
- `datasets`, `faiss-cpu`, `scikit-learn`
- `mergekit`, `safetensors`

## Installation

```bash
conda run -n hiatus-phase3 pip install -e .
```

## Repository Layout

```text
authorship/
  model.py                     # Unified inference API
  data/                        # Data modules + utilities
  models/                      # Embedder/reranker implementations
  training/                    # Training loops, losses, schedulers
  preprocessing/               # BM25 mining, dense mining, corpus curation
    bm25_mining.py             #   BM25 hard-negative mining
    embedder_mining.py         #   Dense hard-pair mining (uses trained embedder)
  evaluation/                  # TA1/TA2 evaluation helpers
  tools/model_merge.py         # Raw checkpoint mergekit wrapper

configs/
  embedder/default.yaml
  reranker/default.yaml
  mergekit/model_merging.template.yaml

scripts/
  preprocess.sh                # BM25 hard-negative mining
  train_embedder.sh            # Embedder (retriever) training
  mine_hard_pairs.sh           # Dense hard-pair mining (requires trained embedder)
  train_reranker.sh            # Reranker training
  evaluate_checkpoints.sh      # Checkpoint watcher + HRS eval + new W&B eval run
  merge_models.sh              # Checkpoint merging
```

---

## Full Pipeline

The system is trained in three stages:

```
Stage 1                 Stage 2                  Stage 3
──────────────          ────────────────────      ──────────────────────────────
BM25 preprocess   →     Train Embedder       →    Dense hard-pair mining
(for embedder)                                    (uses trained embedder)
                                                        ↓
                                                  Train Reranker
                                                  (hard pos + hard neg)
```

---

## Quick Start

### Step 1 — BM25 Hard-Negative Mining (for embedder training)

Mines lexically-similar hard negatives for contrastive embedder training.
Adds `sameAuthor_docIDs` and `hard_negative_docIDs` to the dataset.

```bash
conda run -n hiatus-phase3 bash scripts/preprocess.sh \
    <dataset_name> <output_dir> <languages> <top_k>
```

Example:

```bash
conda run -n hiatus-phase3 bash scripts/preprocess.sh \
    Hieuman/reddit_bm25 ./data/reddit_bm25 en 512
```

Arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `dataset_name` | `Hieuman/reddit_bm25` | HF dataset name or local disk path |
| `output_dir` | `./data/reddit_bm25` | Where to save the updated dataset |
| `languages` | `en` | Space-separated language codes |
| `top_k` | `512` | BM25 candidates to store per document |

---

### Step 2 — Train Embedder

Trains the retriever model on contrastive loss with BM25-mined hard negatives.

```bash
conda run -n hiatus-phase3 bash scripts/train_embedder.sh [hydra_overrides]
```

Defaults to 8 GPUs. Override for single-GPU:

```bash
conda run -n hiatus-phase3 bash scripts/train_embedder.sh training.devices=1
```

Key config knobs (`configs/embedder/default.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `training.devices` | 8 | Number of GPUs |
| `training.epochs` | 2 | Training epochs |
| `data.num_hard_negatives` | 126 | Hard negatives per sample |
| `data.num_train_examples` | 10000 | Cap per dataset |
| `training.resume_checkpoint` | `outputs/merged-8B.v3-eer-wins` | Starting checkpoint |

Checkpoints are written to `outputs/` and logs to `wandb/`.

---

### Step 3 — Dense Hard-Pair Mining (for reranker training)

Uses the **trained embedder** to mine semantically hard examples:

- **`hard_negative_docIDs`** — cross-author pairs with the *highest* cosine
  similarity (most confusable pairs; harder than BM25 negatives).
- **`hard_positive_docIDs`** — same-author pairs with the *lowest* cosine
  similarity (stylistically diverse same-author pairs; hardest to identify).

```bash
bash scripts/mine_hard_pairs.sh \
    [DATASET_NAME] [OUTPUT_DIR] [LANGUAGES] \
    [TOP_K_NEG] [TOP_K_POS] [EMBEDDER_CONFIG_DIR] [EMBEDDING_DIR] [NUM_GPUS]
```

All arguments are optional; defaults are shown in parentheses:

| Arg | Default | Description |
|-----|---------|-------------|
| `DATASET_NAME` | `Hieuman/reddit_bm25` | HF dataset or local path |
| `OUTPUT_DIR` | `./data/reddit_hard_pairs` | Where to save the updated dataset |
| `LANGUAGES` | `en` | Space-separated language codes |
| `TOP_K_NEG` | `512` | Hard negatives to store per document |
| `TOP_K_POS` | `50` | Hard positives to store per document |
| `EMBEDDER_CONFIG_DIR` | `outputs/merged-4B.v4-eer-wins` | Config dir with `config.yaml` + checkpoint |
| `EMBEDDING_DIR` | `./data/embeddings` | Cache dir for shards, memmap, and FAISS index |
| `NUM_GPUS` | `1` | GPUs to use for parallel encoding |

Examples:

```bash
# Single GPU, all defaults
bash scripts/mine_hard_pairs.sh

# 4 GPUs, custom embedder checkpoint
bash scripts/mine_hard_pairs.sh \
    Hieuman/reddit_bm25 ./data/reddit_hard_pairs en \
    512 50 outputs/my-embedder-checkpoint ./data/embeddings 4

# Multiple languages
bash scripts/mine_hard_pairs.sh \
    Hieuman/reddit_bm25 ./data/reddit_hard_pairs 'en zh ar' \
    512 50 outputs/merged-4B.v4-eer-wins ./data/embeddings 4
```

**What the script does internally (five resumable phases):**

| Phase | Runs on | Output |
|-------|---------|--------|
| 1. Encode | All GPUs (parallel, interleaved shards) | `rank{r}_of{ws}.npy` float16 embeddings |
| 2. Merge | Rank 0 | `embeddings.npy` docID-ordered memmap |
| 3. FAISS index | Rank 0 | `index.faiss` (IVFFlat ≤1M docs, IVFPQ >1M) |
| 4. Mine | All GPUs (parallel) | `hard_neg_rank{r}.pkl`, `hard_pos_rank{r}.pkl` |
| 5. Update dataset | Rank 0 | DatasetDict with new columns saved to `OUTPUT_DIR` |

Interrupted runs can be **resumed** — each phase checks for its output file
and skips if already complete.

> **Note on `EMBEDDER_CONFIG_DIR`:** The directory must contain `config.yaml`
> (written automatically by the training script) and a checkpoint file
> (`model.safetensors.index.json`, `model.safetensors`, `model.pt`, or
> `model.ckpt`). The default `outputs/merged-4B.v4-eer-wins` is the standard
> merged checkpoint location.

---

### Step 4 — Train Reranker

Trains the reranker on hard pairs produced in Step 3.
Point `data.dataset_names` at the output directory from Step 3.

```bash
conda run -n hiatus-phase3 bash scripts/train_reranker.sh [hydra_overrides]
```

Defaults to 4 GPUs. Override for single-GPU:

```bash
conda run -n hiatus-phase3 bash scripts/train_reranker.sh training.devices=1
```

To use the dense-mined dataset:

```bash
conda run -n hiatus-phase3 bash scripts/train_reranker.sh \
    "data.dataset_names=[./data/reddit_hard_pairs]" \
    training.devices=1
```

Key config knobs (`configs/reranker/default.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `training.devices` | 4 | Number of GPUs |
| `training.epochs` | 5 | Training epochs |
| `data.num_pos_per_query` | 1 | Hard positives per query |
| `data.num_neg_per_query` | 1 | Hard negatives per query |
| `data.num_train_examples` | 50000 | Cap per dataset |
| `data.global_batch_size` | 64 | Global batch size across all GPUs |

**How the dataset columns are used:**

| Column | Training source | Fallback |
|--------|----------------|---------|
| `hard_positive_docIDs` | same-author pairs, lowest cosine sim first | `sameAuthor_docIDs` (random) |
| `hard_negative_docIDs` | cross-author pairs, highest cosine sim first | `BM25_retrieved_docIDs` |

The reranker dataset loader prefers the dense-mined columns when present and
falls back automatically, so BM25-preprocessed datasets remain compatible.

---

### Step 5 — Merge Checkpoints (optional)

```bash
conda run -n hiatus-phase3 python -m authorship.tools.model_merge \
  --models '["/path/to/model_a.pt", "/path/to/model_b.pt"]' \
  --weights '[0.5, 0.5]' \
  --config-path configs/mergekit/model_merging.yaml \
  --output-dir outputs/merged-model \
  --run-merge
```

Without `--run-merge` the command only prepares and prints the merge command.
Non-safetensors checkpoints are converted to model-only state dicts in a
`prepared/` subdirectory automatically.

---

## Unified Inference API

Primary entry point: `authorship/model.py` (`AuthorshipModel`).

```python
from authorship.model import AuthorshipModel

model = AuthorshipModel(
    embedder_config_path="configs/embedder/default.yaml",
    embedder_checkpoint_path="outputs/merged-8B.v4/model.safetensors.index.json",
    reranker_config_path="configs/reranker/default.yaml",   # optional
    reranker_checkpoint_path=None,                           # optional
)

# Encode texts → numpy array (n, D)
embs = model.encode(["sample text A", "sample text B"])

# Retrieve top-k candidates by cosine similarity
candidates = ["candidate 1", "candidate 2", "candidate 3"]
ret = model.retrieve("query text", candidates, top_k=2)
print(ret["indices"], ret["scores"])

# Rerank: embedder retrieval + reranker score interpolation
ret = model.reranker("query text", candidates, top_k=2)

# Compare two author portfolios → similarity in [0, 1]
score = model.compare(
    ["author1 doc1", "author1 doc2"],
    ["author2 doc1"],
)
print(score)
```

---

## Configuration

All training is Hydra-driven. Override any key from the CLI:

```bash
conda run -n hiatus-phase3 bash scripts/train_embedder.sh \
  training.devices=1 \
  training.epochs=1 \
  data.num_train_examples=5000
```

Important defaults to check before training:

- `training.devices` — defaults to 8 (embedder) or 4 (reranker).
- `training.output_dir` — controls artifact location.
- `training.resume_checkpoint` — may point to an existing checkpoint.

---

## Evaluation

Evaluation code is in `authorship/evaluation/`.

- TA1-style retrieval metrics (e.g., Success@k)
- TA2-style attribution metrics (e.g., EER, FAR, FRR)

`Evaluator` includes fallback implementations if the `hiatus-metrics` library
is unavailable.

For checkpoint-by-checkpoint evaluation (with a fresh W&B eval run), use:

```bash
bash scripts/evaluate_checkpoints.sh \
  --mode embedder \
  --config_dir outputs/embedder-8B-v2 \
  --checkpoint_dir outputs/embedder-8B-v2 \
  --wandb_project authorship-embedder-eval
```

Reranker mode requires embedder paths:

```bash
bash scripts/evaluate_checkpoints.sh \
  --mode reranker \
  --config_dir outputs/reranker-v0 \
  --checkpoint_dir outputs/reranker-v0 \
  --embedder_config_dir outputs/embedder-v0 \
  --embedder_checkpoint_path outputs/embedder-v0/checkpoint_step6000.ckpt \
  --wandb_project authorship-reranker-eval
```

---

## Practical Notes

- Training and merge jobs are compute and storage heavy. Avoid accidental long runs.
- Intermediate files (embeddings, FAISS index, shard pickles) are written to
  `--embedding-dir` during dense mining and can be deleted after the dataset
  is saved.
- All artifacts are under `outputs/` (checkpoints) and `wandb/` (logs).
- Keep experiments config-driven; avoid hardcoding values in code.
