# uoregon-authorship

Authorship attribution and retrieval with a retriever-reranker pipeline.

This repository provides:

- A unified inference API through `AuthorshipModel`.
- Training pipelines for an embedder (retriever) and reranker.
- BM25 hard-negative preprocessing.
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

Using the environment convention in this workspace:

```bash
conda run -n wemg pip install -e .
```

If you are actively using a different environment, replace `wemg` with your env name.

## Repository Layout

```text
authorship/
  model.py                     # Unified inference API
  data/                        # Data modules + utilities
  models/                      # Embedder/reranker implementations
  training/                    # Training loops, losses, schedulers
  preprocessing/               # BM25 mining and corpus curation
  evaluation/                  # TA1/TA2 evaluation helpers
  tools/model_merge.py         # Raw checkpoint mergekit wrapper

configs/
  embedder/default.yaml
  reranker/default.yaml
  mergekit/model_merging.template.yaml

scripts/
  preprocess.sh
  train_embedder.sh
  train_reranker.sh
  merge_models.sh
```

## Quick Start

### 1) Preprocess BM25 Hard Negatives

```bash
conda run -n wemg bash scripts/preprocess.sh <dataset_name> <output_dir> <languages> <top_k>
```

Example:

```bash
conda run -n wemg bash scripts/preprocess.sh Hieuman/reddit_bm25 ./data/reddit_bm25 en 512
```

### 2) Train Embedder

```bash
conda run -n wemg bash scripts/train_embedder.sh [hydra_overrides]
```

Single-GPU example:

```bash
conda run -n wemg bash scripts/train_embedder.sh training.devices=1
```

### 3) Train Reranker

```bash
conda run -n wemg bash scripts/train_reranker.sh [hydra_overrides]
```

Single-GPU example:

```bash
conda run -n wemg bash scripts/train_reranker.sh training.devices=1
```

### 4) Merge Checkpoints

You can merge raw checkpoints (`.pt`, `.ckpt`, `.bin`, `.safetensors`) with:

```bash
conda run -n wemg python -m authorship.tools.model_merge \
  --models '["/path/to/model_a.pt", "/path/to/model_b.pt"]' \
  --weights '[0.5, 0.5]' \
  --config-path configs/mergekit/model_merging.yaml \
  --output-dir outputs/merged-model \
  --run-merge
```

Notes:

- Without `--run-merge`, the command only prepares/prints the merge command.
- Non-safetensors checkpoints are converted to model-only checkpoints in a `prepared/` directory.

## Unified Inference API

Primary entry point: `authorship/model.py` (`AuthorshipModel`).

Supported operations:

- `encode(text_or_list)`
- `retrieve(query, candidates, top_k=...)`
- `reranker(query, candidates, top_k=...)`
- `compare(author1_texts, author2_texts)`

Example:

```python
from authorship.model import AuthorshipModel

model = AuthorshipModel(
    embedder_config_path="configs/embedder/default.yaml",
    embedder_checkpoint_path="outputs/embedder-8B-v2/checkpoint_step1000.ckpt",
    reranker_config_path="configs/reranker/default.yaml",  # optional
    reranker_checkpoint_path=None,                           # optional
)

# Encode texts
embs = model.encode(["sample text A", "sample text B"])

# Retrieve candidates for a query
candidates = ["candidate 1", "candidate 2", "candidate 3"]
ret = model.retrieve("query text", candidates, top_k=2)
print(ret["indices"], ret["scores"].shape)

# Compare two author portfolios (returns similarity in [0, 1])
score = model.compare(
    ["author1 doc1", "author1 doc2"],
    ["author2 doc1", "author2 doc2"],
)
print(score)
```

## Configuration

Hydra configs:

- Embedder: `configs/embedder/default.yaml`
- Reranker: `configs/reranker/default.yaml`

Override values from CLI, for example:

```bash
conda run -n wemg bash scripts/train_embedder.sh \
  training.devices=1 \
  training.epochs=1 \
  data.num_train_examples=5000
```

Important defaults to check before training:

- `training.devices` can be multi-GPU by default.
- `training.output_dir` controls artifact location.
- `training.resume_checkpoint` may point to an existing model.

## Evaluation

Evaluation code is in `authorship/evaluation/`.

- TA1-style retrieval metrics (e.g., S@k)
- TA2-style attribution metrics (e.g., EER)

`Evaluator` includes fallback implementations if `hiatus` metrics are unavailable.

## Practical Notes

- Training and merge jobs can be compute and storage heavy.
- Artifacts are typically written under `outputs/` and logs under `wandb/`.
- Keep experiments config-driven rather than hardcoding values in code.
