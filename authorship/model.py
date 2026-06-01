"""Unified authorship attribution model API.

Provides a single class wrapping both the retriever (embedder) and reranker
with methods for encode, retrieve, reranker, and compare.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from authorship.models.embedder import WrappedEmbeddingModel
from authorship.models.reranker import WrappedRerankerModel

logger = logging.getLogger(__name__)


class AuthorshipModel:
    """Unified API for authorship attribution and retrieval.

    Wraps an embedder (retriever) and optional reranker, providing:
        - encode(text) -> embeddings
        - retrieve(query, candidates) -> top-k results
        - reranker(query, candidates) -> top-k with reranking
        - compare(author1, author2) -> similarity score
    """

    def __init__(
        self,
        embedder_config_path: str,
        embedder_checkpoint_path: Optional[str] = None,
        reranker_config_path: Optional[str] = None,
        reranker_checkpoint_path: Optional[str] = None,
        reranker_weight: float = 0.5,
        embedder_max_length: int = 2048,
        batch_size: int = 32,
        reranker_agg: str = "max",
        reranker_agg_topk: int = 3,
    ):
        self._reranker = None
        self.reranker_weight = reranker_weight
        self.embedder_max_length = embedder_max_length
        self.batch_size = batch_size
        # TA2 author-level aggregation of (q_doc, c_doc) reranker scores:
        #   "max"        — best-matching doc pair (original; one pair can dominate)
        #   "mean"       — average over all doc pairs (robust, dilutes signal)
        #   "topk_mean"  — average of the top-`reranker_agg_topk` pairs (compromise)
        self.reranker_agg = reranker_agg
        self.reranker_agg_topk = reranker_agg_topk

        self._init_embedder(embedder_config_path, embedder_checkpoint_path)
        if reranker_config_path and os.path.exists(reranker_config_path):
            self._init_reranker(reranker_config_path, reranker_checkpoint_path)

    def _init_embedder(self, config_path: str, checkpoint_path: Optional[str]):
        config = OmegaConf.load(config_path)
        self._embedder = WrappedEmbeddingModel(
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

    def _init_reranker(self, config_path: str, checkpoint_path: Optional[str]):
        config = OmegaConf.load(config_path)
        self._reranker = WrappedRerankerModel(
            model_name_or_path=config.model.model_name_or_path,
            use_lora=config.model.lora.use_lora,
            lora_r=config.model.lora.get("r", 16),
            lora_alpha=config.model.lora.get("alpha", 32),
            lora_dropout=config.model.lora.get("dropout", 0.1),
            target_modules=list(config.model.lora.get("target_modules", ["all"])),
            adapter_name=config.model.lora.get("name"),
            quantization=config.model.get("quantization", False),
            attn_implementation=config.model.get("attn_implementation"),
            model_checkpoint=checkpoint_path,
            max_length=config.data.get("max_seq_length", 2048),
            batch_size=config.data.get("global_batch_size", 8),
            instruction=config.data.get("instruction"),
        )
        rr_cfg = getattr(config, "reranking", {})
        self.reranker_weight = float(getattr(rr_cfg, "reranker_weight", self.reranker_weight))
        self.reranker_agg = str(getattr(rr_cfg, "aggregation", self.reranker_agg))
        self.reranker_agg_topk = int(getattr(rr_cfg, "agg_topk", self.reranker_agg_topk))

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            text: A single string or list of strings.

        Returns:
            np.ndarray of shape (1, D) for str input, (n, D) for list input.
        """
        if isinstance(text, str):
            text = [text]
        features = self._embedder.batch_encode(text, max_length=self.embedder_max_length, batch_size=self.batch_size)
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy().astype(np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if np.isnan(features).any():
            features = np.nan_to_num(features)
        return features

    def _encode_author(self, texts: List[str]) -> np.ndarray:
        """Mean-pool embeddings for an author's document portfolio."""
        embs = self.encode(texts)
        return embs.mean(axis=0, keepdims=True)

    def encode_authors(self, portfolios: List[List[str]]) -> np.ndarray:
        """Encode a list of author portfolios, mean-pooling each author's docs.

        Returns:
            np.ndarray of shape (n_authors, D).
        """
        return np.vstack([self._encode_author(docs) for docs in portfolios])

    def score_author_matrix(
        self,
        query_portfolios: List[List[str]],
        candidate_portfolios: List[List[str]],
        top_k: int = 16,
        reranker_weight: Optional[float] = None,
        use_reranker: bool = True,
        query_embeddings: Optional[np.ndarray] = None,
        candidate_embeddings: Optional[np.ndarray] = None,
        aggregation: Optional[str] = None,
        agg_topk: Optional[int] = None,
    ) -> np.ndarray:
        """Return (n_qa, n_ca) similarity matrix for author-level attribution (TA2).

        Retrieves by mean-pooled cosine similarity, then reranks top-k candidate
        authors by aggregating the (q_doc, c_doc) reranker scores if a reranker is
        loaded. ``aggregation`` selects the doc-pair aggregator ("max", "mean", or
        "topk_mean"; default from ``self.reranker_agg``); ``agg_topk`` sets the k for
        "topk_mean". Softer aggregators keep one spurious high-scoring pair from
        dominating an author's score.

        Pass ``query_embeddings`` / ``candidate_embeddings`` (n_authors, D) to reuse
        already-computed author embeddings and skip re-encoding — e.g. when the same
        portfolios are scored both embedder-only and with the reranker.
        """
        w = reranker_weight if reranker_weight is not None else self.reranker_weight
        method = (aggregation or self.reranker_agg).lower()
        topk_agg = agg_topk if agg_topk is not None else self.reranker_agg_topk

        q_embs = query_embeddings if query_embeddings is not None else self.encode_authors(query_portfolios)
        c_embs = candidate_embeddings if candidate_embeddings is not None else self.encode_authors(candidate_portfolios)
        retriever_scores = cosine_similarity(q_embs, c_embs).astype(np.float32)

        if self._reranker is None or not use_reranker:
            return retriever_scores

        # Normalize cosine [-1, 1] to [0, 1] to match reranker P(yes) scale.
        # Non-top-K positions are shifted to [-1, 0] so any reranked top-K
        # author strictly outranks any non-top-K author.
        retriever_norm = (retriever_scores + 1.0) / 2.0
        scores = retriever_norm - 1.0
        portfolio_iter = query_portfolios
        if len(query_portfolios) > 1:
            portfolio_iter = tqdm(
                query_portfolios, desc="TA2 rerank (authors)", unit="query",
            )
        for i, q_docs in enumerate(portfolio_iter):
            top_indices = np.argsort(-retriever_scores[i])[:top_k]

            # Accumulate the (query_doc, candidate_doc) cross-product over *all*
            # top-k candidates into a single batched scoring call. Batching across
            # candidates (rather than one score_pairs call per candidate) fills GPU
            # batches and lets score_pairs sort the whole set by length, which cuts
            # padding waste. group_of[p] records which top-k candidate pair p came
            # from so we can aggregate per-candidate afterwards.
            pair_q: List[str] = []
            pair_c: List[str] = []
            group_of: List[int] = []
            for g, j in enumerate(top_indices):
                c_docs = candidate_portfolios[int(j)]
                for qd in q_docs:
                    for cd in c_docs:
                        pair_q.append(qd)
                        pair_c.append(cd)
                        group_of.append(g)

            rr_agg = np.zeros(len(top_indices), dtype=np.float32)
            if pair_q:
                rr_scores = self._reranker.score_pairs(
                    pair_q,
                    pair_c,
                    desc="TA2 rerank (pairs)",
                    show_progress=len(query_portfolios) == 1 and len(pair_q) > self.batch_size,
                )
                # Aggregate each candidate's doc pairs. Candidates with no pairs
                # (empty portfolio) keep 0.0, matching the prior behaviour.
                rr_agg = self._aggregate_groups(
                    np.asarray(rr_scores, dtype=np.float32),
                    np.asarray(group_of),
                    len(top_indices), method, topk_agg,
                )

            for g, j in enumerate(top_indices):
                scores[i, int(j)] = w * float(rr_agg[g]) + (1.0 - w) * retriever_norm[i, int(j)]

        return scores

    @staticmethod
    def _aggregate_groups(
        scores: np.ndarray, groups: np.ndarray, n_groups: int, method: str, topk: int,
    ) -> np.ndarray:
        """Aggregate per-pair scores into one value per group (candidate author).

        ``method``: "max" (best pair), "mean" (all pairs), or "topk_mean" (mean of
        the top-``topk`` pairs). Reranker scores are P(yes) in [0, 1], so empty
        groups stay at 0.0.
        """
        agg = np.zeros(n_groups, dtype=np.float32)
        if scores.size == 0:
            return agg
        if method == "max":
            np.maximum.at(agg, groups, scores)
            return agg
        if method in ("mean", "topk_mean"):
            if method == "topk_mean":
                # Keep only each group's top-`topk` scores: sort by (group asc,
                # score desc) and mask out within-group ranks >= topk.
                order = np.lexsort((-scores, groups))
                groups, scores = groups[order], scores[order]
                change = np.ones(len(groups), dtype=bool)
                change[1:] = groups[1:] != groups[:-1]
                starts = np.flatnonzero(change)
                grp_start = np.repeat(starts, np.diff(np.append(starts, len(groups))))
                within_rank = np.arange(len(groups)) - grp_start
                keep = within_rank < max(1, int(topk))
                groups, scores = groups[keep], scores[keep]
            sums = np.zeros(n_groups, dtype=np.float64)
            counts = np.zeros(n_groups, dtype=np.int64)
            np.add.at(sums, groups, scores)
            np.add.at(counts, groups, 1)
            nz = counts > 0
            agg[nz] = (sums[nz] / counts[nz]).astype(np.float32)
            return agg
        raise ValueError(f"Unknown TA2 aggregation method: {method!r} (use max/mean/topk_mean)")

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    @staticmethod
    def _is_text_array(arr: np.ndarray) -> bool:
        return arr.dtype.kind in ("U", "S", "O") or np.issubdtype(arr.dtype, np.str_)

    def _to_embeddings(self, data: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """Encode text inputs; pass through numeric embedding matrices unchanged."""
        if isinstance(data, str):
            return self.encode([data])
        if isinstance(data, list):
            return self.encode(data)
        if isinstance(data, np.ndarray):
            if data.ndim == 1 and self._is_text_array(data):
                return self.encode(data.tolist())
            if data.dtype.kind in ("U", "S", "O"):
                return self.encode(data.tolist())
            return data
        return self.encode(list(data))

    def retrieve(
        self,
        query: Union[str, List[str], np.ndarray],
        candidates: Union[List[str], np.ndarray],
        top_k: int = 16,
    ) -> Dict[str, np.ndarray]:
        """Retrieve top-k candidates by cosine similarity.

        Args:
            query: Text(s) or pre-computed embeddings (n_q, D).
            candidates: Text list or pre-computed embeddings (n_c, D).
            top_k: Number of candidates to return.

        Returns:
            Dict with 'indices' (n_q, top_k) and 'scores' (n_q, n_c).
        """
        Q = self._to_embeddings(query)
        C = self._to_embeddings(candidates)

        scores = cosine_similarity(Q, C).astype(np.float32)
        top_k = min(top_k, scores.shape[1])
        indices = np.argsort(-scores, axis=1)[:, :top_k]

        return {"indices": indices, "scores": scores}

    # ------------------------------------------------------------------
    # reranker
    # ------------------------------------------------------------------

    def reranker(
        self,
        query: Union[str, List[str]],
        candidates: List[str],
        top_k: int = 16,
        reranker_weight: Optional[float] = None,
        *,
        query_embeddings: Optional[np.ndarray] = None,
        candidate_embeddings: Optional[np.ndarray] = None,
        progress_desc: Optional[str] = "TA1 rerank",
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Retrieve + rerank candidates.

        First retrieves top_k by embedding similarity, then reranks using
        the reranker model with score interpolation.

        Args:
            query: Query text(s).
            candidates: Candidate texts.
            top_k: Number of candidates to retrieve and rerank.
            reranker_weight: Interpolation weight for reranker scores (0-1).
                Defaults to self.reranker_weight set at init.
            query_embeddings / candidate_embeddings: Optional precomputed
                embeddings (n, D) for the retrieval step, to skip re-encoding when
                the caller already embedded these texts. The texts are still used
                for the reranker scoring pass.

        Returns:
            Dict with 'indices' (n_q, top_k) and 'scores' (n_q, n_c).
        """
        retr_query = query_embeddings if query_embeddings is not None else query
        retr_cands = candidate_embeddings if candidate_embeddings is not None else candidates
        retrieval = self.retrieve(retr_query, retr_cands, top_k)

        if self._reranker is None:
            return retrieval

        if isinstance(query, str):
            query = [query]

        w = reranker_weight if reranker_weight is not None else self.reranker_weight

        reranker_scores = self._reranker.score_matrix(
            query_texts=query,
            candidate_texts=candidates,
            candidate_indices=retrieval["indices"],
            desc=progress_desc,
            show_progress=show_progress,
        )

        # Normalize cosine [-1, 1] to [0, 1] to match reranker P(yes) scale.
        # Non-top-K positions are shifted to [-1, 0] so any top-K candidate
        # outranks any non-top-K candidate — preserving the retriever's
        # in-top-K decision while letting the reranker reorder within it.
        retriever_norm = (retrieval["scores"] + 1.0) / 2.0
        interpolated = retriever_norm - 1.0
        for i, top_indices in enumerate(retrieval["indices"]):
            for j in top_indices:
                interpolated[i, j] = w * reranker_scores[i, j] + (1.0 - w) * retriever_norm[i, j]

        final_indices = np.argsort(-interpolated, axis=1)[:, :top_k]

        return {"indices": final_indices, "scores": interpolated}

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------

    def compare(
        self,
        author1: Union[str, List[str]],
        author2: Union[str, List[str]],
    ) -> float:
        """Compare two authors' writing styles.

        Each author can be a single text or list of texts (portfolio).
        Returns a similarity score in [0, 1].

        Args:
            author1: Single text or list of texts by author 1.
            author2: Single text or list of texts by author 2.

        Returns:
            Float similarity score. Higher = more likely same author.
        """
        if isinstance(author1, str):
            author1 = [author1]
        if isinstance(author2, str):
            author2 = [author2]

        emb1 = self._encode_author(author1)
        emb2 = self._encode_author(author2)
        cos_score = float(cosine_similarity(emb1, emb2)[0, 0])

        if self._reranker is None:
            return (cos_score + 1.0) / 2.0

        pair_q, pair_d = [], []
        for t1 in author1:
            for t2 in author2:
                pair_q.append(t1)
                pair_d.append(t2)

        if pair_q:
            reranker_scores = self._reranker.score_pairs(pair_q, pair_d)
            reranker_score = float(reranker_scores.max())
        else:
            reranker_score = 0.0

        w = self.reranker_weight
        retriever_score = (cos_score + 1.0) / 2.0
        return w * reranker_score + (1.0 - w) * retriever_score
