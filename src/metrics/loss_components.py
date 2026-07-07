from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).float()


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    eroded_y = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    eroded_x = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_x, eroded_y)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def _soft_skeletonize(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    mask = mask.float().clamp(0.0, 1.0)
    skeleton = F.relu(mask - _soft_open(mask))
    for _ in range(max(0, iterations - 1)):
        mask = _soft_erode(mask)
        delta = F.relu(mask - _soft_open(mask))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_dice_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).float()
    probabilities = _flatten_batch(probabilities)
    targets = _flatten_batch(targets)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def soft_cldice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).float()
    targets = targets.float()
    prediction_skeleton = _soft_skeletonize(probabilities, iterations)
    target_skeleton = _soft_skeletonize(targets, iterations)

    prediction_skeleton = _flatten_batch(prediction_skeleton)
    target_skeleton = _flatten_batch(target_skeleton)
    probabilities = _flatten_batch(probabilities)
    targets = _flatten_batch(targets)

    topology_precision = ((prediction_skeleton * targets).sum(dim=1) + smooth) / (
        prediction_skeleton.sum(dim=1) + smooth
    )
    topology_sensitivity = ((target_skeleton * probabilities).sum(dim=1) + smooth) / (
        target_skeleton.sum(dim=1) + smooth
    )
    cldice = (2.0 * topology_precision * topology_sensitivity + smooth) / (
        topology_precision + topology_sensitivity + smooth
    )
    return cldice.mean()


def tversky_index(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).float()
    probabilities = _flatten_batch(probabilities)
    targets = _flatten_batch(targets)
    true_positives = (probabilities * targets).sum(dim=1)
    false_positives = (probabilities * (1.0 - targets)).sum(dim=1)
    false_negatives = ((1.0 - probabilities) * targets).sum(dim=1)
    score = (true_positives + smooth) / (
        true_positives + (alpha * false_positives) + (beta * false_negatives) + smooth
    )
    return score.mean()


def loss_component_metrics(logits: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> dict[str, float]:
    loss_name = str(config.get("name", "")).strip().lower()
    metrics: dict[str, float] = {}
    smooth = float(config.get("smooth", 1e-6))

    if loss_name in {
        "bce_with_logits",
        "bce",
        "binary_cross_entropy_with_logits",
        "bce_dice",
        "bce_dice_cldice",
        "bce_dice_soft_cldice",
        "bcedicecldice",
    }:
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        metrics["bce"] = float(bce.item())
        if "bce_weight" in config:
            metrics["weighted_bce"] = float(bce.item() * float(config.get("bce_weight", 0.0)))

    if loss_name in {
        "bce_dice",
        "bce_dice_cldice",
        "bce_dice_soft_cldice",
        "bcedicecldice",
    }:
        dice = soft_dice_score(logits, targets, smooth=smooth)
        metrics["soft_dice_score"] = float(dice.item())
        if "dice_weight" in config:
            metrics["weighted_soft_dice_loss"] = float((1.0 - dice.item()) * float(config.get("dice_weight", 0.0)))

    if loss_name in {
        "cldice",
        "soft_cldice",
        "softcldice",
        "bce_dice_cldice",
        "bce_dice_soft_cldice",
        "bcedicecldice",
        "tversky_soft_cldice",
        "tversky_softcldice",
    }:
        cldice = soft_cldice_score(
            logits,
            targets,
            iterations=int(config.get("iterations", 3)),
            smooth=float(config.get("cldice_smooth", config.get("smooth", 1.0))),
        )
        metrics["soft_cldice_score"] = float(cldice.item())
        if "soft_cldice_weight" in config:
            metrics["weighted_soft_cldice_loss"] = float(
                (1.0 - cldice.item()) * float(config.get("soft_cldice_weight", 0.0))
            )

    if loss_name in {"tversky", "tversky_soft_cldice", "tversky_softcldice"}:
        tversky = tversky_index(
            logits,
            targets,
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.7)),
            smooth=smooth,
        )
        metrics["tversky_index"] = float(tversky.item())
        if "tversky_weight" in config:
            metrics["weighted_tversky_loss"] = float((1.0 - tversky.item()) * float(config.get("tversky_weight", 0.0)))

    return metrics
