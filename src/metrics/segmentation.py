from __future__ import annotations

import torch


# PyTorch core does not currently ship Dice/IoU segmentation metrics,
# so these remain lightweight project-local implementations.
def _prepare_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).float()


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(tensor.shape[0], -1).float()


def dice_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)

    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(score.mean().item())


def iou_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return float(score.mean().item())


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    return dice_score_from_masks(predictions, targets, smooth=smooth)


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    return iou_score_from_masks(predictions, targets, smooth=smooth)
