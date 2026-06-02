"""Corpus curation for embedder training data.

Implements AUM/EL2N-based filtering, stratification, and diversity-aware
sampling to curate high-quality training data for contrastive learning.
"""

import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def compute_metrics(
    embeddings: np.ndarray,
    author_ids: List[str],
    hard_neg_ids: List[List[int]],
    same_author_ids: List[List[int]],
    temperature: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AUM and EL2N metrics for each sample.

    AUM = mean positive similarity - max negative similarity
    EL2N = 1 - P(selecting any positive from the contrastive distribution)
    """
    n = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-8)

    aum_scores = np.zeros(n, dtype=np.float32)
    el2n_scores = np.zeros(n, dtype=np.float32)

    for i in tqdm(range(n), desc="Computing AUM/EL2N"):
        pos_ids = same_author_ids[i]
        neg_ids = hard_neg_ids[i]
        if not pos_ids or not neg_ids:
            continue

        q = normed[i]
        pos_sims = q @ normed[pos_ids].T
        neg_sims = q @ normed[neg_ids].T

        aum_scores[i] = pos_sims.mean() - neg_sims.max()

        all_sims = np.concatenate([pos_sims, neg_sims]) / temperature
        all_sims -= all_sims.max()
        exp_sims = np.exp(all_sims)
        total = exp_sims.sum()
        p_pos = exp_sims[: len(pos_ids)].sum() / total
        el2n_scores[i] = 1.0 - p_pos

    return aum_scores, el2n_scores


def filter_and_stratify(
    n: int,
    aum_scores: np.ndarray,
    el2n_scores: np.ndarray,
    aum_prune_ratio: float = 0.2,
    hard_ratio: float = 0.3,
    semi_hard_ratio: float = 0.4,
) -> Dict[str, np.ndarray]:
    """Filter noisy samples and stratify by difficulty.

    Returns dict with 'hard', 'semi_hard', 'easy' index arrays.
    """
    aum_threshold = np.percentile(aum_scores, aum_prune_ratio * 100)
    keep_mask = aum_scores >= aum_threshold
    kept_indices = np.where(keep_mask)[0]
    kept_el2n = el2n_scores[kept_indices]

    sorted_idx = np.argsort(-kept_el2n)
    n_kept = len(sorted_idx)
    n_hard = int(n_kept * hard_ratio)
    n_semi = int(n_kept * semi_hard_ratio)

    return {
        "hard": kept_indices[sorted_idx[:n_hard]],
        "semi_hard": kept_indices[sorted_idx[n_hard : n_hard + n_semi]],
        "easy": kept_indices[sorted_idx[n_hard + n_semi :]],
    }


def diversity_check(
    embeddings: np.ndarray,
    easy_indices: np.ndarray,
    n_clusters: int = 50,
    min_samples_per_cluster: int = 5,
) -> np.ndarray:
    """Add easy samples from underrepresented clusters for diversity."""
    if len(easy_indices) == 0:
        return easy_indices

    easy_emb = embeddings[easy_indices]
    norms = np.linalg.norm(easy_emb, axis=1, keepdims=True)
    normed = easy_emb / np.maximum(norms, 1e-8)

    n_clusters = min(n_clusters, len(easy_indices))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    labels = kmeans.fit_predict(normed)

    cluster_counts = np.bincount(labels, minlength=n_clusters)
    underrep = np.where(cluster_counts < min_samples_per_cluster)[0]

    if len(underrep) == 0:
        return easy_indices

    extra = []
    for c in underrep:
        members = easy_indices[labels == c]
        extra.extend(members.tolist())
    return np.unique(np.concatenate([easy_indices, np.array(extra)]))


def mine_dense_hard_negatives(
    embeddings: np.ndarray,
    author_ids: List[str],
    top_k: int = 128,
    exclude_top: int = 64,
) -> List[List[int]]:
    """FAISS inner-product search for dense hard negatives, excluding same-author."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)

    d = normed.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(normed)

    fetch_k = top_k + exclude_top + 1
    _, all_ids = index.search(normed, fetch_k)

    result = []
    for i in range(len(embeddings)):
        aid = author_ids[i]
        neighbors = [int(j) for j in all_ids[i] if j != i and author_ids[j] != aid]
        result.append(neighbors[exclude_top : exclude_top + top_k])
    return result


def curate_corpus(
    embeddings: np.ndarray,
    author_ids: List[str],
    same_author_ids: List[List[int]],
    hard_neg_ids: List[List[int]],
    temperature: float = 0.05,
    aum_prune_ratio: float = 0.2,
    n_diversity_clusters: int = 50,
) -> np.ndarray:
    """Full curation pipeline: filter -> stratify -> diversity -> return selected indices."""
    aum, el2n = compute_metrics(embeddings, author_ids, hard_neg_ids, same_author_ids, temperature)
    strata = filter_and_stratify(len(embeddings), aum, el2n, aum_prune_ratio)

    easy_with_diversity = diversity_check(
        embeddings, strata["easy"], n_clusters=n_diversity_clusters,
    )

    selected = np.concatenate([strata["hard"], strata["semi_hard"], easy_with_diversity])
    return np.unique(selected)
