"""Evaluation metrics for authorship attribution (TA1 and TA2).

TA1 (document-level retrieval): Success@k from rank-based metrics.
TA2 (author-level attribution): Equal Error Rate from score-based metrics.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from hiatus.featurespace.metrics import compute_rank_metrics as _hiatus_rank_metrics
    from hiatus.attribution.metrics import compute_attribution_metrics as _hiatus_attr_metrics
    _HAS_HIATUS = True
except ImportError:
    _HAS_HIATUS = False


class Evaluator:
    """Compute TA1/TA2 metrics from similarity score matrices."""

    def __init__(self, task_type: str = "ta1"):
        self.task_type = task_type

    def evaluate_ta1(
        self,
        similarity_scores: np.ndarray,
        ground_truth_positions: List[List[int]],
        k_values: List[int] = None,
    ) -> Dict:
        """Evaluate document-level retrieval (TA1).

        Args:
            similarity_scores: (n_queries, n_candidates) higher = more similar.
            ground_truth_positions: Per-query list of correct candidate indices.
            k_values: Top-k values for S@k.

        Returns:
            Summary metrics dict from hiatus-metrics.
        """
        if k_values is None:
            k_values = [1, 8]

        if not _HAS_HIATUS:
            return self._evaluate_ta1_fallback(similarity_scores, ground_truth_positions, k_values)

        n_q, n_c = similarity_scores.shape
        raw_ranks = np.array([
            rankdata(-similarity_scores[i], method="min").astype(np.int64)
            for i in range(n_q)
        ])
        all_needle_ranks = [
            [int(raw_ranks[i, p]) for p in gt if p < n_c]
            for i, gt in enumerate(ground_truth_positions)
        ]
        _, summary, _ = _hiatus_rank_metrics(
            all_needle_ranks=pd.Series(all_needle_ranks),
            n_candidates=n_c,
            raw_ranks=raw_ranks,
            top_ks=list(k_values),
            disable_progress_bars=True,
        )
        return summary

    def evaluate_ta2(
        self,
        similarity_scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict:
        """Evaluate author-level attribution (TA2).

        Args:
            similarity_scores: (n_queries, n_candidates) score matrix.
            ground_truth: (n_queries, n_candidates) binary ground truth.

        Returns:
            Summary metrics dict including EER.
        """
        if not _HAS_HIATUS:
            return self._evaluate_ta2_fallback(similarity_scores, ground_truth)

        scores_df = pd.DataFrame(similarity_scores)
        gt_df = pd.DataFrame(ground_truth.astype(np.int64))
        predictions_df = (scores_df >= 0.9).astype(np.int64)
        _, summary, _ = _hiatus_attr_metrics(
            ground_truth=gt_df, scores=scores_df,
            predictions=predictions_df,
            far_target=[0.05], which_metrics="all",
        )
        return summary

    @staticmethod
    def _evaluate_ta1_fallback(scores, gt_positions, k_values):
        """Fallback S@k when hiatus-metrics is not installed."""
        n_q = scores.shape[0]
        results = {}
        for k in k_values:
            hits = 0
            for i in range(n_q):
                top_k_idx = np.argsort(-scores[i])[:k]
                if any(p in top_k_idx for p in gt_positions[i]):
                    hits += 1
            results[f"S@{k}"] = hits / max(n_q, 1)
        return results

    @staticmethod
    def _evaluate_ta2_fallback(scores, ground_truth):
        """Fallback EER when hiatus-metrics is not installed."""
        from sklearn.metrics import roc_curve
        y_true = ground_truth.flatten()
        y_score = scores.flatten()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        return {"EER": float(eer)}
