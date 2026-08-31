# uoregon-authorship

Authorship attribution and retrieval with a retriever–reranker pipeline.
University of Oregon submission for the IARPA **HIATUS** program.

The system answers two questions:

- **TA1 — document retrieval:** given a query document, rank candidate
  documents by likelihood of shared authorship.
- **TA2 — author attribution:** given query and candidate author portfolios,
  score each pair as same-author or not.

A **bi-encoder embedder** retrieves candidates by cosine similarity; an
optional **cross-encoder reranker** rescores the top-k window and the two
scores are interpolated.

## Documentation

| Guide | Contents |
|---|---|
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | BM25 hard-negative mining, dense hard-pair mining, corpus curation |
| [docs/TRAINING.md](docs/TRAINING.md) | Embedder and reranker training, config reference, SLURM jobs |
| [docs/INFERENCE.md](docs/INFERENCE.md) | `AuthorshipModel` unified API — encode / retrieve / rerank / compare |
| [docs/EVALUATION.md](docs/EVALUATION.md) | HRS TA1/TA2 evaluation, checkpoint export, analysis helpers |
| [docs/MODEL_MERGING.md](docs/MODEL_MERGING.md) | Merging raw checkpoints via `mergekit` |

---

## Requirements

- Python 3.12 (the reference `hiatus-phase3` environment; ≥3.10 is supported)
- Linux; GPU required for training, mining, and merging
- Key dependencies (pinned in [`requirements.txt`](requirements.txt) to the
  versions installed in `hiatus-phase3`): `torch`, `transformers`, `lightning`,
  `hydra-core`, `datasets`, `faiss-cpu`, `bm25s`, `mergekit`, `safetensors`

Two dependencies are **optional** and commented out in `requirements.txt`:

- `bitsandbytes` — only for 4-bit quantized training (`model.quantization=true`).
- `hiatus` — the restricted HIATUS metrics library. Evaluation falls back to
  local S@k/EER implementations without it; see
  [docs/EVALUATION.md](docs/EVALUATION.md#metrics-backend).

## Installation

All commands run inside the `hiatus-phase3` conda environment:

```bash
conda run -n hiatus-phase3 pip install -e .
```

## Data access

Training datasets are private HuggingFace datasets under the `Hieuman/*`
namespace (listed in `configs/embedder/default.yaml`).

**HRS evaluation data is restricted IARPA HIATUS-program data** and is not
included in this repository. Its filesystem roots are configurable via
environment variables — see
[docs/EVALUATION.md § Restricted data access](docs/EVALUATION.md#restricted-data-access).

```bash
export HRS_DATA_ROOT=/path/to/HRS
export HRS1_DATA_ROOT=/path/to/HRS/TA1
```

---

## Full Pipeline

The system is trained in four stages; merging is optional.

```
Stage 1                Stage 2               Stage 3                    Stage 4
─────────────────      ────────────────      ──────────────────────     ────────────────
BM25 preprocess    →   Train Embedder    →   Dense hard-pair mining  →  Train Reranker
(lexical negatives)    (retriever)           (uses trained embedder)    (hard pos + neg)
                                                                              ↓
                                                                         Evaluate / Merge
```

| Stage | Command | Guide |
|---|---|---|
| 1. BM25 mining | `bash scripts/preprocess.sh` | [PREPROCESSING](docs/PREPROCESSING.md#1-bm25-hard-negative-mining) |
| 2. Train embedder | `bash scripts/train_embedder.sh` | [TRAINING](docs/TRAINING.md#1-train-embedder) |
| 3. Dense mining | `bash scripts/mine_hard_pairs.sh` | [PREPROCESSING](docs/PREPROCESSING.md#2-dense-hard-pair-mining) |
| 4. Train reranker | `bash scripts/train_reranker.sh` | [TRAINING](docs/TRAINING.md#2-train-reranker) |
| Evaluate | `bash scripts/evaluate_single.sh` | [EVALUATION](docs/EVALUATION.md) |
| Merge (optional) | `python -m authorship.tools.model_merge` | [MODEL_MERGING](docs/MODEL_MERGING.md) |

### Quick start

```bash
# 1. Mine BM25 hard negatives
conda run -n hiatus-phase3 bash scripts/preprocess.sh \
    Hieuman/reddit_bm25 ./data/reddit_bm25 en 512

# 2. Train the embedder (single GPU)
conda run -n hiatus-phase3 bash scripts/train_embedder.sh training.devices=1

# 3. Mine dense hard pairs with the trained embedder
bash scripts/mine_hard_pairs.sh Hieuman/reddit_bm25 ./data/reddit_hard_pairs en \
    512 50 outputs/embedder-8B-v2 ./data/embeddings 1

# 4. Train the reranker on those pairs
conda run -n hiatus-phase3 bash scripts/train_reranker.sh \
    "data.dataset_names=[./data/reddit_hard_pairs]" training.devices=1
```

### Inference

```python
from authorship import AuthorshipModel

model = AuthorshipModel(
    embedder_config_path="configs/embedder/default.yaml",
    embedder_checkpoint_path="outputs/merged-8B.v4/model.safetensors.index.json",
)

model.encode(["sample text A", "sample text B"])          # (2, D) embeddings
model.retrieve("query text", candidates, top_k=8)         # TA1 retrieval
model.compare(["author1 doc1", "author1 doc2"], ["author2 doc1"])  # TA2 score in [0, 1]
```

Full API in [docs/INFERENCE.md](docs/INFERENCE.md).

---

## Repository Layout

```text
authorship/
  model.py                     # Unified inference API (AuthorshipModel)
  models/                      # Embedder / reranker / bidirectional adapters
  data/                        # Contrastive + pairwise datasets, samplers
  training/                    # Training loops, losses, GradCache, schedulers
  preprocessing/
    bm25_mining.py             #   BM25 hard-negative mining
    embedder_mining.py         #   Dense hard-pair mining (needs trained embedder)
    corpus_curation.py         #   AUM/EL2N curation utilities (standalone)
  evaluation/                  # HRS TA1/TA2 evaluation
  tools/model_merge.py         # mergekit wrapper for raw checkpoints

configs/
  embedder/default.yaml        # Embedder training config
  reranker/default.yaml        # Reranker training config
  reranker/local_h100.yaml     # Single-node H100 variant
  mergekit/model_merging.template.yaml

scripts/
  preprocess.sh                # Stage 1 — BM25 mining
  train_embedder.sh            # Stage 2 — embedder training
  mine_hard_pairs.sh           # Stage 3 — dense hard-pair mining
  train_reranker.sh            # Stage 4 — reranker training
  evaluate_single.sh           # Evaluate one checkpoint
  evaluate_checkpoints.sh      # Watch a run and evaluate new checkpoints
  evaluate_reranker.sh         # Evaluate all reranker checkpoints (multi-GPU)
  sweep_reranker_weight.sh     # Sweep the interpolation weight
  export_reranker_fabric_ckpt.py  # Consolidate sharded FSDP checkpoints
  summarize_hrs_eval.py        # Reprint eval comparison tables
  diagnose_reranker_pyes.py    # Reranker P(yes) distribution diagnostic
  merge_models.sh              # Checkpoint merging example
  sbatch_*.sh                  # SLURM array/batch job variants
```

## Architecture Notes

- **Embedder** (`models/embedder.py`) — any transformer backbone with pooling,
  optional LoRA, 4-bit quantization, and bidirectional attention.
- **Bidirectional adapters** (`models/bidirectional.py`) — when using a causal
  LM as an encoder, these wrappers replace the causal mask with a padding-only
  mask (Qwen2/3, Mistral, Gemma3). Required for `model.is_bidirectional=true`.
- **Reranker** (`models/reranker.py`) — scores `(query, candidate)` pairs as
  `P(yes)` from an instruction-prompted causal LM.
- **GradCache** (`training/gradcache.py`) — splits large contrastive batches
  into chunks, computes loss over the full batch, then replays per-chunk
  gradients. Makes large effective batch sizes fit in GPU memory.
- **Training** — Lightning Fabric + FSDP, with checkpoint resumption.

## Practical Notes

- Training, mining, and merging are compute- and storage-heavy. Check config
  defaults (`training.devices`, `training.resume_checkpoint`,
  `training.output_dir`) before launching to avoid accidental long runs.
- Multi-GPU defaults: embedder and reranker configs both ship with
  `training.devices=4`. Override with `training.devices=1` for local runs.
- Keep experiments config-driven — override via Hydra rather than editing code.
- Reranker training checkpoints are sharded and must be consolidated before
  inference; see [docs/EVALUATION.md](docs/EVALUATION.md#checkpoint-export).
- Artifacts: checkpoints in `outputs/`, logs in `wandb/`, eval output in
  `eval_results/`. Dense-mining intermediates under `--embedding-dir` are a
  cache and can be deleted after the dataset is written.
