"""Loss functions for embedder and reranker training."""

import torch
from pytorch_metric_learning import losses, miners, distances
from pytorch_metric_learning.utils import distributed as pml_dist


def compute_contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    use_miner: bool = False,
    use_cross_device: bool = True,
    temperature: float = 0.05,
    normalize: bool = True,
) -> torch.Tensor:
    """SupCon loss with optional MultiSimilarity mining and cross-device support."""
    distance = distances.CosineSimilarity() if normalize else distances.LpDistance(normalize_embeddings=False)
    loss_fn = losses.SupConLoss(temperature=temperature, distance=distance)

    if use_cross_device:
        loss_fn = pml_dist.DistributedLossWrapper(loss_fn)

    if use_miner:
        miner = miners.MultiSimilarityMiner(epsilon=0.2, distance=distance)
        if use_cross_device:
            miner = pml_dist.DistributedMinerWrapper(miner)
        hard_pairs = miner(projection, labels)
        return loss_fn(projection, labels, hard_pairs)

    return loss_fn(projection, labels)


def compute_reranker_loss(
    true_logits: torch.Tensor,
    false_logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE loss on P(yes) for reranker training."""
    stacked = torch.stack([false_logits, true_logits], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    probs_yes = log_probs[:, 1].exp()

    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

    return torch.nn.functional.binary_cross_entropy(probs_yes, labels)


def compute_reranker_metrics(
    true_logits: torch.Tensor,
    false_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Compute accuracy, precision, recall, F1 for reranker predictions."""
    stacked = torch.stack([false_logits, true_logits], dim=1)
    probs_yes = torch.softmax(stacked, dim=1)[:, 1]
    preds = (probs_yes >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    pos_mask = labels == 1
    neg_mask = labels == 0

    return {
        "accuracy": (preds == labels).float().mean().item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "avg_p_yes_pos": probs_yes[pos_mask].mean().item() if pos_mask.any() else 0,
        "avg_p_yes_neg": probs_yes[neg_mask].mean().item() if neg_mask.any() else 0,
    }
