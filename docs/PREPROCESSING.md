# Preprocessing

Two mining stages produce the hard positive/negative columns the training
pipelines consume. See the [full pipeline diagram](../README.md#full-pipeline)
for how these fit between the two training stages.

- **BM25 hard-negative mining** — lexical, fast, no trained model required.
  Feeds embedder training.
- **Dense hard-pair mining** — semantic, uses a trained embedder to mine
  harder pairs than BM25 can. Feeds reranker training.

Both stages read/write [HuggingFace `datasets`](https://huggingface.co/docs/datasets)
`DatasetDict`s (a HF Hub name like `Hieuman/reddit_bm25`, or a local path saved
with `save_to_disk`). Required input columns: `docID`, `authorIDs`, `fullText`,
`sameAuthor_docIDs`.

---

## 1. BM25 Hard-Negative Mining

`authorship/preprocessing/bm25_mining.py`, wrapped by `scripts/preprocess.sh`.

Mines lexically-similar hard negatives per document with [`bm25s`](https://github.com/xhluca/bm25s)
and adds two columns:

| Column | Description |
|---|---|
| `sameAuthor_docIDs` | Same-author positives (added if not already present). |
| `hard_negative_docIDs` | Top-k lexically-similar cross-author documents, most-similar first. Filtered by a similarity threshold relative to the max BM25 score. |

> **Legacy column name.** Older datasets store these negatives as
> `BM25_retrieved_docIDs`. The embedder dataset loader
> (`authorship/data/embedder_dataset.py`) auto-renames
> `BM25_retrieved_docIDs` → `hard_negative_docIDs` when the latter is absent,
> and the reranker loader falls back to it directly — so both layouts work
> without extra flags.

```bash
conda run -n hiatus-phase3 bash scripts/preprocess.sh \
    <dataset_name> <output_dir> <languages> <top_k>
```

| Arg | Default | Description |
|---|---|---|
| `dataset_name` | `Hieuman/reddit_bm25` | HF dataset name or local disk path |
| `output_dir` | `./data/reddit_bm25` | Where to save the updated dataset |
| `languages` | `en` | Space-separated language codes |
| `top_k` | `512` | BM25 candidates to store per document |

Example:

```bash
conda run -n hiatus-phase3 bash scripts/preprocess.sh \
    Hieuman/reddit_bm25 ./data/reddit_bm25 en 512
```

### SLURM: all datasets at once

`scripts/sbatch_preprocess.sh` is a SLURM array job that runs BM25 mining
over every dataset in the project's dataset list (one array index per
dataset; missing language splits are skipped).

```bash
sbatch scripts/sbatch_preprocess.sh
# override:
TOP_K=256 OUT_ROOT=./data/bm25 sbatch scripts/sbatch_preprocess.sh
```

Array size (`--array=0-25%10`, i.e. 26 datasets, 10 concurrent) is hardcoded
to match the `DATASETS` array in the script — update both together if you
add/remove datasets. Requires `logs/` (auto-created) and runs on the
`computelong` partition, 16 CPUs / 128G mem, up to 3 days per task.

---

## 2. Dense Hard-Pair Mining

`authorship/preprocessing/embedder_mining.py`, wrapped by `scripts/mine_hard_pairs.sh`.
Requires a **trained embedder checkpoint** (output of
[Training Stage 1](TRAINING.md#1-train-embedder)) or a HuggingFace embedding
model.

Encodes every document with the embedder, builds a FAISS index, and mines:

| Column | Meaning |
|---|---|
| `hard_negative_docIDs` | Cross-author pairs with the **highest** cosine similarity — the most confusable negatives, harder than BM25's lexical negatives. |
| `hard_positive_docIDs` | Same-author pairs with the **lowest** cosine similarity — stylistically diverse same-author pairs, hardest to identify as a match. |

```bash
bash scripts/mine_hard_pairs.sh \
    [DATASET_NAME] [OUTPUT_DIR] [LANGUAGES] \
    [TOP_K_NEG] [TOP_K_POS] [EMBEDDER_CONFIG_DIR] [EMBEDDING_DIR] [NUM_GPUS] \
    [NLIST] [NPROBE] [NEG_BUFFER]
```

All arguments are optional (positional or same-named env vars); defaults:

| Arg | Default | Description |
|---|---|---|
| `DATASET_NAME` | `Hieuman/reddit_bm25` | HF dataset or local path |
| `OUTPUT_DIR` | `./data/reddit_hard_pairs` | Where to save the updated dataset |
| `LANGUAGES` | `en` | Space-separated language codes |
| `TOP_K_NEG` | `512` | Hard negatives to store per document |
| `TOP_K_POS` | `50` | Hard positives to store per document |
| `EMBEDDER_CONFIG_DIR` | `outputs/merged-4B.v4-eer-wins` | Dir with `config.yaml` + checkpoint |
| `EMBEDDING_DIR` | `./data/embeddings` | Cache dir for shards, memmap, FAISS index |
| `NUM_GPUS` | `1` | GPUs for parallel encoding/mining |
| `NLIST` | `65536` | FAISS IVF cluster count (auto-clamped to `N // 39`) |
| `NPROBE` | `128` | FAISS IVF probe count (recall/speed trade-off) |
| `NEG_BUFFER` | `4` | Over-fetch factor: `fetch_k = TOP_K_NEG * NEG_BUFFER`, filtered down to same-author-excluded top-k |

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

# Use a HuggingFace embedding model instead of a trained authorship checkpoint
HF_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B \
    bash scripts/mine_hard_pairs.sh Hieuman/reddit_bm25 ./data/reddit_hard_qwen en

# Wider over-fetch buffer for prolific-author corpora (more candidates to
# filter same-author docs out of before taking the top-k negatives)
NEG_BUFFER=16 bash scripts/mine_hard_pairs.sh
```

`HF_EMBEDDER_MODEL` accepts any Hub sentence/embedding model; pair with
`HF_EMBEDDER_INSTRUCT` (instruction prefix) and `HF_EMBEDDER_ATTN`
(attention implementation, e.g. `flash_attention_2`) as needed.

**What the script does internally (five resumable phases):**

| Phase | Runs on | Output |
|---|---|---|
| 1. Encode | All GPUs (parallel, interleaved shards) | `rank{r}_of{ws}.npy` float16 embeddings |
| 2. Merge | Rank 0 | `embeddings.npy` docID-ordered memmap |
| 3. FAISS index | Rank 0 | `index.faiss` (IVFFlat ≤1M docs, IVFPQ >1M) |
| 4. Mine | All GPUs (parallel) | `hard_neg_rank{r}.pkl`, `hard_pos_rank{r}.pkl` |
| 5. Update dataset | Rank 0 | `DatasetDict` with new columns saved to `OUTPUT_DIR` |

Interrupted runs **resume** automatically — each phase checks for its output
file and skips if already complete.

> **`EMBEDDER_CONFIG_DIR`** must contain `config.yaml` (written automatically
> by `train_embedder.sh`) and a checkpoint file (`model.safetensors.index.json`
> + `model.safetensors`, or `model.pt`, or `model.ckpt`).

Intermediate files under `EMBEDDING_DIR` (shards, memmap, FAISS index) can be
deleted once the dataset is saved to `OUTPUT_DIR` — they're only a cache.

### SLURM: all datasets at once

`scripts/sbatch_mine_hard_pairs.sh` mirrors `sbatch_preprocess.sh` for the
dense-mining stage: one SLURM array task per dataset, 4x 80GB GPUs each.

```bash
sbatch scripts/sbatch_mine_hard_pairs.sh
EMBEDDER_CONFIG_DIR=outputs/my-model sbatch scripts/sbatch_mine_hard_pairs.sh
HF_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B sbatch scripts/sbatch_mine_hard_pairs.sh
```

Datasets in the array are commented/uncommented individually in the script —
check `DATASETS=(...)` before submitting; only uncommented entries run.

---

## 3. Corpus Curation (standalone utility)

`authorship/preprocessing/corpus_curation.py` implements AUM/EL2N-based
quality filtering and diversity-aware stratified sampling for contrastive
training data. It is **not wired into a CLI script** — no `scripts/*.sh`
calls it today. Import and use it directly if you need to curate a dataset:

```python
from authorship.preprocessing.corpus_curation import compute_metrics
# see the module docstring/function signatures for the full API:
#   compute_metrics(embeddings, author_ids, hard_neg_ids, same_author_ids, temperature)
#   -> (aum_scores, el2n_scores)
```

- **AUM** = mean positive similarity − max negative similarity (higher = easier/cleaner sample).
- **EL2N** = 1 − P(selecting any positive from the contrastive distribution) (higher = harder/noisier sample).

Use these scores to filter mislabeled/ambiguous documents or to build
difficulty-stratified training splits before running the mining stages above.
