"""Dense hard-pair mining for reranker training.

Encodes all documents with the trained embedder model, builds a FAISS ANN
index, then mines:

  - ``hard_negative_docIDs``: cross-author document pairs with the HIGHEST
    cosine similarity (most confusable negatives for the reranker).
  - ``hard_positive_docIDs``: same-author document pairs with the LOWEST
    cosine similarity (hardest positives — stylistically diverse same-author
    pairs that the reranker must learn to identify).

These two new columns replace the BM25-mined ``hard_negative_docIDs`` and
random sampling from ``sameAuthor_docIDs`` used in reranker training.

The pipeline has five phases per language split:

  1. **Encode** — each GPU encodes its interleaved shard; outputs are
     L2-normalised so inner-product == cosine similarity.
  2. **Merge** — rank 0 merges per-rank shards into a docID-ordered float16
     memory-mapped file.
  3. **Index** — rank 0 builds and saves a FAISS IVF index
     (``IndexIVFFlat`` for ≤1 M docs, ``IndexIVFPQ`` for larger splits).
  4. **Mine** — each GPU mines hard neg/pos for its shard in parallel.
  5. **Update** — rank 0 merges mining results and saves the updated
     ``DatasetDict``.

All intermediate files are written to ``--embedding-dir``. Completed phases
are detected by the presence of their output files, so interrupted runs can
be resumed cheaply.

Usage (multi-GPU via torchrun):
    torchrun --nproc-per-node=4 \\
        -m authorship.preprocessing.embedder_mining \\
        --dataset-name Hieuman/reddit_bm25 \\
        --output-dir ./data/reddit_hard_pairs \\
        --languages en \\
        --embedder-config-dir outputs/merged-4B.v4-eer-wins

Usage (single GPU):
    python -m authorship.preprocessing.embedder_mining \\
        --dataset-name Hieuman/reddit_bm25 \\
        --output-dir ./data/reddit_hard_pairs \\
        --languages en \\
        --embedder-config-dir outputs/merged-4B.v4-eer-wins

See ``scripts/mine_hard_pairs.sh`` for a convenience wrapper.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from authorship.models.embedder import WrappedEmbeddingModel
from authorship.preprocessing.bm25_mining import add_column_with_polars, build_same_author_ids


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_dist() -> Tuple[int, int, int]:
    """Initialise torch.distributed when launched via torchrun.

    Returns:
        (rank, world_size, local_rank).  All three are 0/1 for single-process
        runs (no ``RANK`` env var).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Embedder loading
# ---------------------------------------------------------------------------

def load_embedder(
    config_dir: str,
    checkpoint_path: Optional[str] = None,
) -> WrappedEmbeddingModel:
    """Load a ``WrappedEmbeddingModel`` from *config_dir*.

    Follows the same pattern as ``AuthorshipModel._init_embedder`` in
    ``authorship/model.py``.

    Args:
        config_dir: Directory that contains ``config.yaml`` and (optionally)
            the model checkpoint.  Default checkpoint search order:
            ``model.safetensors.index.json``, ``model.safetensors``,
            ``model.pt``, ``model.ckpt``.
        checkpoint_path: Explicit checkpoint path; overrides auto-detection.

    Raises:
        FileNotFoundError: If ``config.yaml`` is not found in *config_dir*.
    """
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {config_dir!r}. "
            "Pass a directory that contains config.yaml and the model checkpoint."
        )
    config = OmegaConf.load(config_path)

    if checkpoint_path is None:
        for fname in [
            "model.safetensors.index.json",
            "model.safetensors",
            "model.pt",
            "model.ckpt",
        ]:
            cand = os.path.join(config_dir, fname)
            if os.path.exists(cand):
                checkpoint_path = cand
                break
        if checkpoint_path is None:
            # WrappedEmbeddingModel accepts a directory with an index.json
            checkpoint_path = config_dir

    return WrappedEmbeddingModel(
        model_name_or_path=config.model.model_name_or_path,
        use_lora=config.model.lora.use_lora,
        dropout_prob=config.model.get("dropout", 0.1),
        lora_r=config.model.lora.get("r", 16),
        lora_alpha=config.model.lora.get("alpha", 32),
        lora_dropout=config.model.lora.get("dropout", 0.1),
        target_modules=list(config.model.lora.get("target_modules", ["all"])),
        adapter_name=config.model.lora.get("name"),
        quantization=config.model.get("quantization", False),
        attn_implementation=config.model.get("attn_implementation"),
        pooling_method=config.model.get("pooling", "mean"),
        is_bidirectional=config.model.get("is_bidirectional", False),
        model_checkpoint=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Encode
# ---------------------------------------------------------------------------

def encode_shard(
    model: WrappedEmbeddingModel,
    texts: List[str],
    doc_ids: List[int],
    shard_path: str,
    ids_path: str,
    batch_size: int,
    max_length: int,
    rank: int,
) -> None:
    """Encode *texts* for this rank's shard and save as float16 ``npy`` files.

    Embeddings are L2-normalised so inner-product equals cosine similarity.
    Skips encoding if both output files already exist (resumable).

    Args:
        model: Loaded ``WrappedEmbeddingModel``.
        texts: Document texts for this rank's shard (in shard order).
        doc_ids: Corresponding global document IDs.
        shard_path: Path for the ``float16`` embedding array ``(S, D)``.
        ids_path: Path for the global docID array ``(S,)``.
        batch_size: Texts per forward pass.
        max_length: Maximum token length for encoding.
        rank: This process's rank (for logging).
    """
    if os.path.exists(shard_path) and os.path.exists(ids_path):
        print(f"[Rank {rank}] Shard already exists, skipping encode.")
        return

    all_reps: List[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"[Rank {rank}] Encoding",
        disable=not is_main(),
    ):
        batch = texts[start : start + batch_size]
        reps = model.encode(batch, max_length=max_length)  # (B, D) float32
        reps = F.normalize(reps, p=2, dim=1)
        all_reps.append(reps.cpu().numpy().astype(np.float16))

    arr = np.concatenate(all_reps, axis=0)  # (S, D) float16
    np.save(shard_path, arr)
    np.save(ids_path, np.array(doc_ids, dtype=np.int64))
    print(f"[Rank {rank}] Saved shard {arr.shape} → {shard_path}")


# ---------------------------------------------------------------------------
# Phase 2 — Merge shards into a docID-ordered memmap
# ---------------------------------------------------------------------------

def merge_shards(
    shard_dir: str,
    world_size: int,
    N: int,
) -> Tuple[str, int]:
    """Merge per-rank embedding shards into one float16 memory-mapped file.

    Each shard holds embeddings for docIDs ``rank, rank+ws, rank+2*ws, …``.
    The merge writes each row directly into the correct docID position so the
    resulting file is indexed as ``embeddings[docID]``.

    Args:
        shard_dir: Directory containing per-rank shard files.
        world_size: Number of ranks.
        N: Total number of documents.

    Returns:
        ``(merged_emb_path, D)`` — path to the merged memmap and embedding dim.
    """
    merged_emb_path = os.path.join(shard_dir, "embeddings.npy")
    if os.path.exists(merged_emb_path):
        print("[Rank 0] Merged embeddings already exist, skipping merge.")
        peek = np.load(
            os.path.join(shard_dir, f"rank0_of{world_size}.npy"), mmap_mode="r"
        )
        return merged_emb_path, peek.shape[1]

    peek = np.load(
        os.path.join(shard_dir, f"rank0_of{world_size}.npy"), mmap_mode="r"
    )
    D = peek.shape[1]
    del peek

    merged = np.memmap(merged_emb_path, dtype="float16", mode="w+", shape=(N, D))

    CHUNK = 10_000  # rows per write — keeps RAM bounded
    for r in range(world_size):
        shard_emb = np.load(
            os.path.join(shard_dir, f"rank{r}_of{world_size}.npy"), mmap_mode="r"
        )
        rank_ids = np.load(
            os.path.join(shard_dir, f"rank{r}_of{world_size}_ids.npy")
        )
        for start in range(0, len(rank_ids), CHUNK):
            sl = slice(start, start + CHUNK)
            merged[rank_ids[sl]] = np.asarray(shard_emb[sl])  # force page-in
        print(f"[Rank 0] Merged shard {r}: {shard_emb.shape}")

    merged.flush()
    print(f"[Rank 0] Merged embeddings: ({N}, {D}) → {merged_emb_path}")
    return merged_emb_path, D


# ---------------------------------------------------------------------------
# Phase 3 — Build FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(
    merged_emb_path: str,
    N: int,
    D: int,
    index_path: str,
    nlist: int,
) -> None:
    """Build and save a FAISS IVF index over the merged embeddings.

    * ``IndexIVFFlat``  for N ≤ 1 M (exact search within cells).
    * ``IndexIVFPQ``    for N  > 1 M (compressed; ~64 bytes/vector for D=4096).

    Skips if *index_path* already exists.

    Args:
        merged_emb_path: Path to the float16 memmap ``(N, D)``.
        N: Total number of vectors.
        D: Embedding dimension.
        index_path: Where to write the serialised FAISS index.
        nlist: Number of IVF centroids (rule of thumb: ``sqrt(N)``).
    """
    import faiss

    if os.path.exists(index_path):
        print("[Rank 0] FAISS index already exists, skipping build.")
        return

    all_emb = np.memmap(merged_emb_path, dtype="float16", mode="r", shape=(N, D))
    quantizer = faiss.IndexFlatIP(D)

    if N > 1_000_000:
        M = min(64, D // 4)  # PQ subvectors — 64 bytes/vec for D≥256
        index = faiss.IndexIVFPQ(quantizer, D, nlist, M, 8)
        print(f"[Rank 0] Using IndexIVFPQ (N={N:,}, nlist={nlist}, M={M})")
    else:
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"[Rank 0] Using IndexIVFFlat (N={N:,}, nlist={nlist})")

    n_train = min(max(nlist * 39, 100_000), N)
    sample_idx = np.random.choice(N, n_train, replace=False)
    train_data = np.asarray(all_emb[sample_idx]).astype(np.float32)
    print(f"[Rank 0] Training FAISS index on {n_train:,} vectors …")
    index.train(train_data)
    del train_data

    CHUNK = 100_000
    for start in tqdm(range(0, N, CHUNK), desc="[Rank 0] Building index"):
        chunk = np.asarray(all_emb[start : start + CHUNK]).astype(np.float32)
        index.add(chunk)

    faiss.write_index(index, index_path)
    print(f"[Rank 0] FAISS index ({index.ntotal:,} vectors) → {index_path}")


# ---------------------------------------------------------------------------
# Phase 4 — Mine hard pairs
# ---------------------------------------------------------------------------

def mine_pairs_for_shard(
    rank_doc_ids: List[int],
    all_embeddings: np.ndarray,
    author_ids: List[str],
    same_author_ids: List[List[int]],
    index,  # faiss.Index
    top_k_neg: int,
    top_k_pos: int,
    rank: int,
    neg_buffer: int = 4,
    batch_size: int = 256,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Mine hard negatives and hard positives for this rank's document shard.

    **Hard negatives** — FAISS ANN search returns the highest-scoring
    cross-author neighbours (most confusable, sorted descending by cosine sim).

    **Hard positives** — exact inner-product over ``sameAuthor_docIDs``
    identifies same-author pairs with the *lowest* cosine similarity
    (hardest to identify as same-author, sorted ascending by cosine sim).

    Args:
        rank_doc_ids: Global docIDs handled by this rank.
        all_embeddings: Float16 memmap ``(N, D)`` indexed by global docID.
        author_ids: ``author_ids[docID]`` → author string.
        same_author_ids: ``same_author_ids[docID]`` → list of same-author docIDs.
        index: Loaded FAISS index (``index.nprobe`` should be set by caller).
        top_k_neg: Hard negatives to keep per query.
        top_k_pos: Hard positives to keep per query.
        rank: This process's rank (for logging).
        neg_buffer: Over-fetch factor for FAISS search before same-author
            filtering (``fetch_k = top_k_neg × neg_buffer``).
        batch_size: Queries per FAISS search call.

    Returns:
        ``(hard_neg_dict, hard_pos_dict)`` keyed by global docID.
    """
    hard_neg_dict: Dict[int, List[int]] = {}
    hard_pos_dict: Dict[int, List[int]] = {}

    for start in tqdm(
        range(0, len(rank_doc_ids), batch_size),
        desc=f"[Rank {rank}] Mining",
        disable=not is_main(),
    ):
        batch_ids = rank_doc_ids[start : start + batch_size]
        query_embs = np.asarray(all_embeddings[batch_ids]).astype(np.float32)  # (B, D)

        # ── Hard negatives (FAISS ANN) ───────────────────────────────────────
        fetch_k = min(top_k_neg * neg_buffer, index.ntotal)
        _, faiss_hits = index.search(query_embs, fetch_k)  # (B, fetch_k)

        for i, gid in enumerate(batch_ids):
            author = author_ids[gid]
            cross_author = [
                int(h)
                for h in faiss_hits[i]
                if h >= 0 and h != gid and author_ids[h] != author
            ]
            hard_neg_dict[gid] = cross_author[:top_k_neg]

        # ── Hard positives (exact cosine sim over sameAuthor_docIDs) ─────────
        for i, gid in enumerate(batch_ids):
            pos_ids = same_author_ids[gid]
            if not pos_ids:
                hard_pos_dict[gid] = []
                continue

            pos_embs = np.asarray(all_embeddings[pos_ids]).astype(np.float32)  # (P, D)
            q_emb = query_embs[i : i + 1]  # (1, D)
            sims = (q_emb @ pos_embs.T)[0]  # (P,) — valid because both are L2-normalised

            # Ascending: lowest cosine sim first = hardest positives
            order = np.argsort(sims)
            hard_pos_dict[gid] = [int(pos_ids[j]) for j in order[:top_k_pos]]

    return hard_neg_dict, hard_pos_dict


# ---------------------------------------------------------------------------
# Top-level pipeline per language split
# ---------------------------------------------------------------------------

def process_language_split(
    dataset: datasets.Dataset,
    lang: str,
    model: WrappedEmbeddingModel,
    embedding_dir: str,
    top_k_neg: int,
    top_k_pos: int,
    batch_size: int,
    max_length: int,
    nlist: int,
    nprobe: int,
    rank: int,
    world_size: int,
    dataset_key: str,
) -> datasets.Dataset:
    """Run the full mining pipeline for one language split.

    All ranks participate in Phases 1 and 4.  Phases 2, 3, and 5 run on
    rank 0 only, with ``barrier()`` calls for synchronisation.

    Args:
        dataset: The HF ``Dataset`` for this language split.
        lang: Language code (used for directory naming and logging).
        model: Loaded ``WrappedEmbeddingModel`` (on this rank's GPU).
        embedding_dir: Root cache directory.
        top_k_neg: Hard negatives per query.
        top_k_pos: Hard positives per query.
        batch_size: Encoding/mining batch size.
        max_length: Max token length for encoding.
        nlist: FAISS IVF nlist.
        nprobe: FAISS search probes.
        rank: This process's rank.
        world_size: Total number of processes.
        dataset_key: Sanitised dataset name used for sub-directory naming.

    Returns:
        The updated ``Dataset`` with ``hard_negative_docIDs`` and
        ``hard_positive_docIDs`` columns added (meaningful on rank 0 only).
    """
    import faiss

    shard_dir = os.path.join(embedding_dir, f"{dataset_key}_{lang}")
    if is_main():
        os.makedirs(shard_dir, exist_ok=True)
    barrier()

    N = len(dataset)
    texts = dataset["fullText"]

    # Ensure sameAuthor_docIDs exists — each rank builds independently (cheap groupby)
    if "sameAuthor_docIDs" not in dataset.column_names:
        print(f"[Rank {rank}] Building sameAuthor_docIDs for {lang} …")
        dataset = build_same_author_ids(dataset)

    # ── Phase 1: Encode ───────────────────────────────────────────────────────
    my_doc_ids = list(range(rank, N, world_size))  # interleaved sharding
    my_texts = [texts[i] for i in my_doc_ids]
    shard_path = os.path.join(shard_dir, f"rank{rank}_of{world_size}.npy")
    ids_path = os.path.join(shard_dir, f"rank{rank}_of{world_size}_ids.npy")
    encode_shard(model, my_texts, my_doc_ids, shard_path, ids_path, batch_size, max_length, rank)

    barrier()  # All ranks finish encoding before rank 0 merges

    # ── Phase 2: Merge (rank 0 only) ─────────────────────────────────────────
    merged_emb_path = os.path.join(shard_dir, "embeddings.npy")
    index_path = os.path.join(shard_dir, "index.faiss")
    D: int

    if is_main():
        merged_emb_path, D = merge_shards(shard_dir, world_size, N)

    barrier()

    # ── Phase 3: Build FAISS index (rank 0 only) ──────────────────────────────
    if is_main():
        build_faiss_index(merged_emb_path, N, D, index_path, nlist)

    barrier()  # Others wait until index file is ready

    # ── Load merged embeddings + index (all ranks) ───────────────────────────
    peek = np.load(
        os.path.join(shard_dir, f"rank0_of{world_size}.npy"), mmap_mode="r"
    )
    D = peek.shape[1]
    del peek

    all_emb = np.memmap(merged_emb_path, dtype="float16", mode="r", shape=(N, D))
    index = faiss.read_index(index_path)
    index.nprobe = nprobe

    author_ids: List[str] = dataset["authorIDs"]
    same_author_ids: List[List[int]] = dataset["sameAuthor_docIDs"]

    # ── Phase 4: Mine (all ranks) ─────────────────────────────────────────────
    hard_neg_path = os.path.join(shard_dir, f"hard_neg_rank{rank}.pkl")
    hard_pos_path = os.path.join(shard_dir, f"hard_pos_rank{rank}.pkl")

    if os.path.exists(hard_neg_path) and os.path.exists(hard_pos_path):
        print(f"[Rank {rank}] Mining results already exist, skipping.")
    else:
        hard_neg_dict, hard_pos_dict = mine_pairs_for_shard(
            rank_doc_ids=my_doc_ids,
            all_embeddings=all_emb,
            author_ids=author_ids,
            same_author_ids=same_author_ids,
            index=index,
            top_k_neg=top_k_neg,
            top_k_pos=top_k_pos,
            rank=rank,
            batch_size=batch_size,
        )
        with open(hard_neg_path, "wb") as fh:
            pickle.dump(hard_neg_dict, fh, protocol=4)
        with open(hard_pos_path, "wb") as fh:
            pickle.dump(hard_pos_dict, fh, protocol=4)

    barrier()

    # ── Phase 5: Merge results + update dataset (rank 0 only) ────────────────
    if is_main():
        merged_neg: Dict[int, List[int]] = {}
        merged_pos: Dict[int, List[int]] = {}
        for r in range(world_size):
            with open(os.path.join(shard_dir, f"hard_neg_rank{r}.pkl"), "rb") as fh:
                merged_neg.update(pickle.load(fh))
            with open(os.path.join(shard_dir, f"hard_pos_rank{r}.pkl"), "rb") as fh:
                merged_pos.update(pickle.load(fh))

        hard_neg_col = [merged_neg[i] for i in range(N)]
        hard_pos_col = [merged_pos[i] for i in range(N)]

        dataset = add_column_with_polars(dataset, "hard_negative_docIDs", hard_neg_col)
        dataset = add_column_with_polars(dataset, "hard_positive_docIDs", hard_pos_col)

        avg_neg = np.mean([len(x) for x in hard_neg_col])
        avg_pos = np.mean([len(x) for x in hard_pos_col])
        print(
            f"[{lang}] Mining complete. "
            f"Avg hard negatives/doc: {avg_neg:.1f}, "
            f"avg hard positives/doc: {avg_pos:.1f}"
        )

    barrier()
    return dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine dense hard pairs (positives + negatives) for reranker training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="HF dataset name or local disk path (DatasetDict with language splits)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to save the updated DatasetDict with new hard-pair columns",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en"],
        help="Language splits to process",
    )
    parser.add_argument(
        "--embedder-config-dir", type=str, default="outputs/merged-4B.v4-eer-wins",
        help="Directory containing config.yaml and the embedder checkpoint",
    )
    parser.add_argument(
        "--embedder-checkpoint-path", type=str, default=None,
        help="Explicit checkpoint path; overrides auto-detection from config dir",
    )
    parser.add_argument(
        "--top-k-neg", type=int, default=512,
        help="Number of hard negatives to store per query document",
    )
    parser.add_argument(
        "--top-k-pos", type=int, default=50,
        help="Number of hard positives to store per query document",
    )
    parser.add_argument(
        "--embedding-dir", type=str, default="./data/embeddings",
        help="Cache directory for shard files, merged embeddings, and FAISS index",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Encoding/mining batch size per GPU",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max token length for document encoding",
    )
    parser.add_argument(
        "--nlist", type=int, default=65536,
        help="FAISS IVF nlist centroids (rule of thumb: sqrt(N))",
    )
    parser.add_argument(
        "--nprobe", type=int, default=128,
        help="FAISS search probes at query time (higher = better recall, slower)",
    )
    args = parser.parse_args()

    rank, world_size, _ = init_dist()

    if is_main():
        os.makedirs(args.embedding_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    barrier()

    print(f"[Rank {rank}/{world_size}] Loading embedder from {args.embedder_config_dir!r} …")
    model = load_embedder(args.embedder_config_dir, args.embedder_checkpoint_path)

    if os.path.exists(args.dataset_name):
        hf_data = datasets.load_from_disk(args.dataset_name)
    else:
        hf_data = datasets.load_dataset(args.dataset_name)

    # Sanitise dataset name for use as a directory component
    dataset_key = Path(args.dataset_name).name.replace("/", "_").replace("\\", "_")

    result: Dict[str, datasets.Dataset] = {}
    for lang in args.languages:
        if lang not in hf_data:
            print(f"[Rank {rank}] Skipping {lang!r}: not in dataset")
            continue

        print(f"[Rank {rank}] Processing {lang!r}: {len(hf_data[lang]):,} documents")
        ds = process_language_split(
            dataset=hf_data[lang],
            lang=lang,
            model=model,
            embedding_dir=args.embedding_dir,
            top_k_neg=args.top_k_neg,
            top_k_pos=args.top_k_pos,
            batch_size=args.batch_size,
            max_length=args.max_length,
            nlist=args.nlist,
            nprobe=args.nprobe,
            rank=rank,
            world_size=world_size,
            dataset_key=dataset_key,
        )
        if is_main():
            result[lang] = ds

    if is_main():
        print(f"Saving updated dataset to {args.output_dir!r} …")
        datasets.DatasetDict(result).save_to_disk(args.output_dir)
        print("Done.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
