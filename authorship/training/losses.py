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
    group_ids: torch.Tensor | None = None,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 0.0,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCE loss plus optional within-query pairwise ranking loss.

    Uses binary_cross_entropy_with_logits on (true - false) logit difference,
    which is equivalent to softmax-over-2 + BCE but numerically stable for
    extreme logit values via the log-sum-exp trick.

    ``pos_weight`` scales the positive-class term of the BCE; set it to
    ``num_neg / num_pos`` to counteract a pos:neg imbalance (e.g. 4.0 for a
    3:12 pos:neg sampling) and keep the operating point centered.
    """
    logit_diff = true_logits - false_logits  # log-odds = log P(yes)/P(no)
    target_labels = labels

    if label_smoothing > 0:
        target_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

    pw = None
    if pos_weight and pos_weight != 1.0:
        pw = torch.tensor(pos_weight, device=logit_diff.device, dtype=logit_diff.dtype)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit_diff, target_labels.to(logit_diff.device), pos_weight=pw
    )
    if ranking_loss_weight <= 0 or group_ids is None:
        return bce_loss

    ranking_loss = compute_pairwise_ranking_loss(
        logit_diff,
        labels.to(logit_diff.device),
        group_ids.to(logit_diff.device),
        margin=ranking_margin,
    )
    return bce_loss + ranking_loss_weight * ranking_loss


def compute_pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Soft pairwise loss requiring positives to outrank negatives per query."""
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        group_scores = scores[mask]
        group_labels = labels[mask]
        pos_scores = group_scores[group_labels == 1]
        neg_scores = group_scores[group_labels == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        deltas = pos_scores[:, None] - neg_scores[None, :]
        losses.append(torch.nn.functional.softplus(margin - deltas).mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def compute_reranker_metrics(
    true_logits: torch.Tensor,
    false_logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> dict:
    """Compute accuracy, precision, recall, F1 for reranker predictions."""
    stacked = torch.stack([false_logits, true_logits], dim=1)
    probs_yes = torch.softmax(stacked, dim=1)[:, 1]
    logit_diff = true_logits - false_logits
    preds = (probs_yes >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    pos_mask = labels == 1
    neg_mask = labels == 0
    avg_p_yes_pos = probs_yes[pos_mask].mean().item() if pos_mask.any() else 0
    avg_p_yes_neg = probs_yes[neg_mask].mean().item() if neg_mask.any() else 0

    metrics = {
        "accuracy": (preds == labels).float().mean().item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "label_pos_rate": labels.float().mean().item(),
        "pred_yes_rate": preds.float().mean().item(),
        "avg_p_yes_pos": avg_p_yes_pos,
        "avg_p_yes_neg": avg_p_yes_neg,
        "p_yes_gap": avg_p_yes_pos - avg_p_yes_neg,
    }
    if group_ids is not None:
        ranking_loss = compute_pairwise_ranking_loss(
            logit_diff,
            labels.to(logit_diff.device),
            group_ids.to(logit_diff.device),
        )
        metrics["ranking_loss"] = ranking_loss.item()
        metrics["pairwise_accuracy"] = compute_pairwise_accuracy(
            logit_diff,
            labels.to(logit_diff.device),
            group_ids.to(logit_diff.device),
        )
    return metrics


def compute_pairwise_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
) -> float:
    """Fraction of within-query positive/negative pairs ranked correctly."""
    correct = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        group_scores = scores[mask]
        group_labels = labels[mask]
        pos_scores = group_scores[group_labels == 1]
        neg_scores = group_scores[group_labels == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        correct.append((pos_scores[:, None] > neg_scores[None, :]).float().mean())
    if not correct:
        return 0.0
    return torch.stack(correct).mean().item()
