# Inference

`AuthorshipModel` (`authorship/model.py`) is the single entry point for
inference. It wraps a trained **embedder** (retriever) and an **optional
reranker**, exposing document-level retrieval (TA1) and author-level
attribution (TA2).

```python
from authorship import AuthorshipModel   # or: from authorship.model import AuthorshipModel
```

## Loading

```python
model = AuthorshipModel(
    embedder_config_path="configs/embedder/default.yaml",
    embedder_checkpoint_path="outputs/merged-8B.v4/model.safetensors.index.json",
    reranker_config_path="configs/reranker/default.yaml",   # optional
    reranker_checkpoint_path="outputs/reranker-v1/exported/checkpoint_step10000.pt",  # optional
)
```

| Parameter | Default | Description |
|---|---|---|
| `embedder_config_path` | *(required)* | Training config YAML — model architecture is rebuilt from it |
| `embedder_checkpoint_path` | `None` | Trained weights; base model is used if omitted |
| `reranker_config_path` | `None` | Enables the reranker. Silently ignored if the path doesn't exist |
| `reranker_checkpoint_path` | `None` | Reranker weights |
| `reranker_weight` | `0.5` | Interpolation weight `w` in `w * P(yes) + (1-w) * cos_norm` |
| `embedder_max_length` | `2048` | Token truncation length for encoding |
| `batch_size` | `32` | Encoding batch size |
| `reranker_agg` | `"max"` | TA2 doc-pair aggregation: `max`, `mean`, `topk_mean` |
| `reranker_agg_topk` | `3` | `k` used when `reranker_agg="topk_mean"` |

If the reranker config contains a `reranking:` section, its `reranker_weight`,
`aggregation`, and `agg_topk` values **override** the constructor arguments.

**Supported checkpoint formats** (auto-detected): `.ckpt` (Lightning, reads the
`"model"` key), `.pt` (flat state dict), `.safetensors`,
`.safetensors.index.json` (sharded), or a **directory** containing
`model.safetensors.index.json`. Sharded reranker training checkpoints must be
consolidated first — see
[EVALUATION.md § Checkpoint export](EVALUATION.md#checkpoint-export).

---

## `encode` — text → embeddings

```python
embs = model.encode(["sample text A", "sample text B"])   # (2, D) float32
emb  = model.encode("a single document")                  # (1, D)
```

Returns a NumPy array; `NaN`s are zeroed defensively.

## `retrieve` — TA1 document retrieval

Ranks candidates by cosine similarity.

```python
candidates = ["candidate 1", "candidate 2", "candidate 3"]
ret = model.retrieve("query text", candidates, top_k=2)
ret["indices"]   # (n_queries, top_k) candidate indices, best first
ret["scores"]    # (n_queries, n_candidates) full cosine score matrix
```

Both `query` and `candidates` accept raw text **or pre-computed embeddings**
(`np.ndarray`, numeric dtype), so you can encode once and retrieve many times.

## `reranker` — TA1 retrieval + reranking

Retrieves `top_k` by embedding similarity, then rescores that window with the
cross-encoder and interpolates:

```python
ret = model.reranker("query text", candidates, top_k=16, reranker_weight=0.5)
```

Falls back to plain `retrieve` if no reranker is loaded. Pass
`query_embeddings=` / `candidate_embeddings=` to reuse embeddings for the
retrieval step (the raw texts are still needed for reranker scoring).

> **Scoring convention.** Cosine scores are normalized from `[-1, 1]` to
> `[0, 1]` to match the reranker's `P(yes)` scale. Candidates *outside* the
> top-k window are shifted into `[-1, 0]`, so any reranked candidate strictly
> outranks any non-top-k candidate. This preserves the retriever's in-top-k
> decision while letting the reranker reorder within the window — which means
> **`top_k` caps recall**: the reranker can never recover a true match the
> retriever left outside the window. Widen `top_k` if retriever recall@top_k
> is the bottleneck.

## `compare` — author portfolio similarity

```python
score = model.compare(
    ["author1 doc1", "author1 doc2"],   # str or list of str
    ["author2 doc1"],
)
```

Returns a float in `[0, 1]`; higher = more likely the same author. Each
portfolio is mean-pooled into a single author embedding. With a reranker
loaded, the result is `w * max P(yes) over all doc pairs + (1-w) * cos_norm`.

## `score_author_matrix` — TA2 attribution at scale

The batched author-level equivalent of `compare`, used by the HRS TA2
evaluation path:

```python
scores = model.score_author_matrix(
    query_portfolios,        # List[List[str]] — one list of docs per query author
    candidate_portfolios,    # List[List[str]]
    top_k=16,
    aggregation="topk_mean", # "max" | "mean" | "topk_mean"
    agg_topk=16,
)                            # -> (n_query_authors, n_candidate_authors)
```

Retrieves by mean-pooled cosine similarity, then reranks the top-k candidate
authors by aggregating every `(query_doc, candidate_doc)` reranker score.

**Choosing an aggregator** — this materially affects TA2 EER:

| Method | Behavior |
|---|---|
| `max` | Best-matching document pair. Sensitive: one spurious high-scoring pair can carry an author. |
| `mean` | Average over all pairs. Robust, but dilutes a genuine strong match. |
| `topk_mean` | Mean of the top-`agg_topk` pairs. Compromise between the two. |

Pass `query_embeddings=` / `candidate_embeddings=` (shape `(n_authors, D)`) to
skip re-encoding — useful when scoring the same portfolios both embedder-only
and with the reranker. Set `use_reranker=False` for an embedder-only baseline.

---

## Related helpers

- `model.encode_authors(portfolios)` — mean-pooled author embeddings, `(n_authors, D)`.
- Auxiliary embedder implementations live in `authorship/models/`:
  `embedder.py` (main, LoRA/quantization/bidirectional-aware),
  `hf_embedder.py` (plain HuggingFace embedding models, used by dense mining),
  `reranker.py` (`P(yes)` cross-encoder), `bidirectional.py` (causal→bidirectional attention adapters).
