from __future__ import annotations

import torch


def _prepare_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).float()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(score.mean().item())


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return float(score.mean().item())

