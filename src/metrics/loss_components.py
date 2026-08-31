from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.metrics.segmentation import soft_cldice_scores_from_probabilities


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).float()


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
    iterations: int | torch.Tensor = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).float()
    return soft_cldice_scores_from_probabilities(
        probabilities,
        targets.float(),
        iterations=iterations,
        smooth=smooth,
    ).mean()


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


def loss_component_metrics(
    logits: torch.Tensor, targets: torch.Tensor, config: dict[str, Any],
    geometry_weights: torch.Tensor | None = None,
    soft_cldice_iterations: torch.Tensor | None = None,
) -> dict[str, float]:
    loss_name = str(config.get("name", "")).strip().lower()
    metrics: dict[str, float] = {}
    smooth = float(config.get("smooth", 1e-6))

    if loss_name == "multiclass_geometry_ce_dice_loci_cldice":
        from src.losses.combined import MulticlassGeometryCEDiceLociCLDiceLoss

        loss = MulticlassGeometryCEDiceLociCLDiceLoss(
            geometry_aware_ce_weight=float(config.get("geometry_aware_ce_weight", 0.25)),
            dice_weight=float(config.get("dice_weight", 0.55)),
            soft_cldice_weight=float(config.get("soft_cldice_weight", 0.20)),
            iterations=int(config.get("iterations", 30)),
            smooth=smooth,
            cldice_smooth=float(config.get("cldice_smooth", 1.0)),
        )
        parts = loss.components(
            logits, targets, geometry_weights, soft_cldice_iterations
        )
        return {
            "geometry_aware_cross_entropy": float(parts["geometry_aware_ce"].item()),
            "multiclass_dice_loss": float(parts["dice"].item()),
            "loci_soft_cldice_loss": float(parts["loci_cldice"].item()),
        }

    if loss_name == "multiclass_ce_dice_loci_cldice":
        from src.losses.combined import MulticlassCEDiceLociCLDiceLoss

        loss = MulticlassCEDiceLociCLDiceLoss(
            cross_entropy_weight=float(config.get("cross_entropy_weight", 0.2)),
            dice_weight=float(config.get("dice_weight", 0.5)),
            loci_cldice_weight=float(config.get("loci_cldice_weight", 0.3)),
            iterations=int(config.get("iterations", 30)),
            smooth=smooth,
            cldice_smooth=float(config.get("cldice_smooth", 1.0)),
        )
        parts = loss.components(logits, targets, soft_cldice_iterations)
        return {
            "cross_entropy": float(parts["cross_entropy"].item()),
            "multiclass_dice_loss": float(parts["dice"].item()),
            "loci_soft_cldice_loss": float(parts["loci_cldice"].item()),
        }

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
            iterations=(
                int(config.get("iterations", 3))
                if soft_cldice_iterations is None
                else soft_cldice_iterations
            ),
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
