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
    listwise_loss_weight: float = 0.0,
    listwise_temperature: float = 1.0,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCE plus optional within-query pairwise and listwise ranking terms.

    Uses binary_cross_entropy_with_logits on (true - false) logit difference,
    which is equivalent to softmax-over-2 + BCE but numerically stable.

    BCE provides cross-query calibration (anchors P(yes) for EER and the
    score interpolation in AuthorshipModel.reranker()). The listwise term
    (ListNet-style multi-positive softmax CE over each group) provides the
    within-query ranking signal aligned with S@k.

    ``pos_weight`` scales the positive-class term of the BCE; leave at 1.0
    when the listwise term is active (listwise handles imbalance natively).
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
    total = bce_loss

    if group_ids is not None:
        gids = group_ids.to(logit_diff.device)
        lbls = labels.to(logit_diff.device)
        if ranking_loss_weight > 0:
            total = total + ranking_loss_weight * compute_pairwise_ranking_loss(
                logit_diff, lbls, gids, margin=ranking_margin,
            )
        if listwise_loss_weight > 0:
            total = total + listwise_loss_weight * compute_listwise_loss(
                logit_diff, lbls, gids, temperature=listwise_temperature,
            )
    return total


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


def compute_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-group multi-positive softmax cross-entropy (ListNet top-1).

    For each query group, the target distribution is uniform over the
    positives; the loss is cross-entropy (equivalently KL up to a constant)
    between that target and softmax(scores / temperature). Lower temperature
    sharpens focus on the hardest negative; higher smooths the gradient.
    """
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        s = scores[mask] / temperature
        y = labels[mask].to(s.dtype)
        n_pos = y.sum()
        if n_pos == 0 or n_pos == y.numel():
            continue  # need at least one positive AND one negative
        target = y / n_pos
        log_p = torch.log_softmax(s, dim=0)
        losses.append(-(target * log_p).sum())
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
        gids = group_ids.to(logit_diff.device)
        lbls = labels.to(logit_diff.device)
        ranking_loss = compute_pairwise_ranking_loss(logit_diff, lbls, gids)
        listwise_loss = compute_listwise_loss(logit_diff, lbls, gids, temperature=1.0)
        metrics["ranking_loss"] = ranking_loss.item()
        metrics["listwise_loss"] = listwise_loss.item()
        metrics["pairwise_accuracy"] = compute_pairwise_accuracy(logit_diff, lbls, gids)
        metrics["rank1_acc"] = compute_rankk_accuracy(logit_diff, lbls, gids, k=1)
        metrics["rank3_acc"] = compute_rankk_accuracy(logit_diff, lbls, gids, k=3)
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


def compute_rankk_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    k: int = 1,
) -> float:
    """Fraction of query groups where at least one positive ranks in top-k by score.

    Aligns with the S@k eval objective: per-group hit rate when the reranker
    pushes a same-author candidate into the top-k of its candidate list.
    """
    hits = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        group_scores = scores[mask]
        group_labels = labels[mask]
        if (group_labels == 1).sum() == 0 or (group_labels == 0).sum() == 0:
            continue
        kk = min(k, group_scores.numel())
        topk_idx = torch.topk(group_scores, kk).indices
        hits.append((group_labels[topk_idx] == 1).any().float())
    if not hits:
        return 0.0
    return torch.stack(hits).mean().item()
