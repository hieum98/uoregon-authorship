"""Unified authorship attribution model API.

Provides a single class wrapping both the retriever (embedder) and reranker
with methods for encode, retrieve, reranker, and compare.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
    ):
        self._reranker = None
        self.reranker_weight = reranker_weight
        self.embedder_max_length = embedder_max_length

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
        self.reranker_weight = float(
            getattr(getattr(config, "reranking", {}), "reranker_weight", self.reranker_weight)
        )

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
        features = self._embedder.batch_encode(text, max_length=self.embedder_max_length, batch_size=32)
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy().astype(np.float32)
        if features.ndim == 1:
            features = features[None, :]
        return features

    def _encode_author(self, texts: List[str]) -> np.ndarray:
        """Mean-pool embeddings for an author's document portfolio."""
        embs = self.encode(texts)
        return embs.mean(axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

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
        if isinstance(query, (str, list)) and not isinstance(query, np.ndarray):
            if isinstance(query, str):
                query = [query]
            Q = self.encode(query)
        else:
            Q = query

        if isinstance(candidates, list) and not isinstance(candidates, np.ndarray):
            C = self.encode(candidates)
        else:
            C = candidates

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
    ) -> Dict[str, np.ndarray]:
        """Retrieve + rerank candidates.

        First retrieves top_k by embedding similarity, then reranks using
        the reranker model with score interpolation.

        Args:
            query: Query text(s).
            candidates: Candidate texts.
            top_k: Number of candidates to consider.

        Returns:
            Dict with 'indices' (n_q, top_k) and 'scores' (n_q, n_c).
        """
        retrieval = self.retrieve(query, candidates, top_k)

        if self._reranker is None:
            return retrieval

        if isinstance(query, str):
            query = [query]

        reranker_scores = self._reranker.score_matrix(
            query_texts=query,
            candidate_texts=candidates,
            candidate_indices=retrieval["indices"],
        )

        w = self.reranker_weight
        interpolated = w * reranker_scores + (1.0 - w) * retrieval["scores"]
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
