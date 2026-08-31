from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.metrics.segmentation import soft_cldice_scores_from_probabilities


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).float()


def _sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).float()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


class BCEDiceSoftCLDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.6,
        soft_cldice_weight: float = 0.1,
        iterations: int = 5,
        smooth: float = 1e-6,
        cldice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.soft_cldice_weight = soft_cldice_weight
        self.bce_dice = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
        )
        self.soft_cldice = SoftCLDiceLoss(
            iterations=iterations,
            smooth=cldice_smooth,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce_dice_loss = self.bce_dice(logits, targets)
        soft_cldice_loss = self.soft_cldice(
            logits, targets, soft_cldice_iterations=soft_cldice_iterations
        )
        return bce_dice_loss + (self.soft_cldice_weight * soft_cldice_loss)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = _sigmoid_logits(logits)
        probabilities = _flatten_batch(probabilities)
        targets = _flatten_batch(targets)

        true_positives = (probabilities * targets).sum(dim=1)
        false_positives = (probabilities * (1.0 - targets)).sum(dim=1)
        false_negatives = ((1.0 - probabilities) * targets).sum(dim=1)

        tversky_index = (true_positives + self.smooth) / (
            true_positives
            + (self.alpha * false_positives)
            + (self.beta * false_negatives)
            + self.smooth
        )
        return 1.0 - tversky_index.mean()


class CLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = _sigmoid_logits(logits)
        cldice_score = soft_cldice_scores_from_probabilities(
            probabilities,
            targets,
            iterations=(
                self.iterations
                if soft_cldice_iterations is None
                else soft_cldice_iterations
            ),
            smooth=self.smooth,
        )
        return 1.0 - cldice_score.mean()


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = _sigmoid_logits(logits)
        cldice_score = soft_cldice_scores_from_probabilities(
            probabilities,
            targets,
            iterations=(
                self.iterations
                if soft_cldice_iterations is None
                else soft_cldice_iterations
            ),
            smooth=self.smooth,
        )
        return 1.0 - cldice_score.mean()


class TverskySoftCLDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        tversky_weight: float = 0.7,
        soft_cldice_weight: float = 0.3,
        iterations: int = 3,
        smooth: float = 1e-6,
        cldice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.tversky_weight = tversky_weight
        self.soft_cldice_weight = soft_cldice_weight
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.soft_cldice = SoftCLDiceLoss(iterations=iterations, smooth=cldice_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        soft_cldice_loss = self.soft_cldice(
            logits, targets, soft_cldice_iterations=soft_cldice_iterations
        )
        return (self.tversky_weight * tversky_loss) + (self.soft_cldice_weight * soft_cldice_loss)


def soft_cldice_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int | torch.Tensor = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Differentiable clDice loss for probabilities that are already normalized."""
    return 1.0 - soft_cldice_scores_from_probabilities(
        probabilities, targets, iterations=iterations, smooth=smooth
    ).mean()


class MulticlassCEDiceLociCLDiceLoss(nn.Module):
    def __init__(
        self,
        cross_entropy_weight: float = 0.2,
        dice_weight: float = 0.5,
        loci_cldice_weight: float = 0.3,
        iterations: int = 30,
        smooth: float = 1e-6,
        cldice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.loci_cldice_weight = loci_cldice_weight
        self.iterations = iterations
        self.smooth = smooth
        self.cldice_smooth = cldice_smooth

    def _dice_loss(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = _flatten_batch(probabilities)
        targets = _flatten_batch(targets)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()

    def components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Autocast promotes cross-entropy during the training loss, but diagnostics
        # call this method outside autocast. A large FP16 reduction can overflow
        # even for finite logits, so keep all multiclass loss components in FP32.
        logits = logits.float()
        targets = targets.long()
        ce = F.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        p_loci = probabilities[:, 1:2]
        p_inoculum = probabilities[:, 2:3]
        y_loci = (targets == 1).float().unsqueeze(1)
        y_inoculum = (targets == 2).float().unsqueeze(1)
        dice = 0.5 * (
            self._dice_loss(p_loci, y_loci)
            + self._dice_loss(p_inoculum, y_inoculum)
        )
        loci_cldice = soft_cldice_from_probabilities(
            p_loci,
            y_loci,
            iterations=(
                self.iterations
                if soft_cldice_iterations is None
                else soft_cldice_iterations
            ),
            smooth=self.cldice_smooth,
        )
        return {"cross_entropy": ce, "dice": dice, "loci_cldice": loci_cldice}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = self.components(logits, targets, soft_cldice_iterations)
        return (
            self.cross_entropy_weight * parts["cross_entropy"]
            + self.dice_weight * parts["dice"]
            + self.loci_cldice_weight * parts["loci_cldice"]
        )


class MulticlassGeometryCEDiceLociCLDiceLoss(MulticlassCEDiceLociCLDiceLoss):
    def __init__(
        self,
        geometry_aware_ce_weight: float = 0.25,
        dice_weight: float = 0.55,
        soft_cldice_weight: float = 0.20,
        iterations: int = 30,
        smooth: float = 1e-6,
        cldice_smooth: float = 1.0,
    ) -> None:
        super().__init__(
            cross_entropy_weight=0.0,
            dice_weight=dice_weight,
            loci_cldice_weight=soft_cldice_weight,
            iterations=iterations,
            smooth=smooth,
            cldice_smooth=cldice_smooth,
        )
        self.geometry_aware_ce_weight = geometry_aware_ce_weight
        self.soft_cldice_weight = soft_cldice_weight

    def components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        geometry_weights: torch.Tensor | None = None,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if geometry_weights is None:
            raise ValueError("Geometry-aware cross-entropy requires geometry weights.")
        logits = logits.float()
        targets = targets.long()
        geometry_weights = geometry_weights.float()
        if geometry_weights.ndim == 4 and geometry_weights.shape[1] == 1:
            geometry_weights = geometry_weights.squeeze(1)
        if geometry_weights.shape != targets.shape:
            raise ValueError(
                "Geometry weight shape must match target shape: "
                f"weights={tuple(geometry_weights.shape)}, targets={tuple(targets.shape)}."
            )
        if not bool(torch.isfinite(geometry_weights).all().item()):
            raise ValueError("Geometry weights must be finite.")
        if bool((geometry_weights < 1.0).any().item()):
            raise ValueError("Geometry weights must not reduce any pixel below 1.")

        pixel_ce = F.cross_entropy(logits, targets, reduction="none")
        geometry_ce = (geometry_weights * pixel_ce).sum() / geometry_weights.sum().clamp_min(
            1.0e-6
        )
        probabilities = torch.softmax(logits, dim=1)
        p_loci = probabilities[:, 1:2]
        p_inoculum = probabilities[:, 2:3]
        y_loci = (targets == 1).float().unsqueeze(1)
        y_inoculum = (targets == 2).float().unsqueeze(1)
        dice = 0.5 * (
            self._dice_loss(p_loci, y_loci)
            + self._dice_loss(p_inoculum, y_inoculum)
        )
        loci_cldice = soft_cldice_from_probabilities(
            p_loci,
            y_loci,
            iterations=(
                self.iterations
                if soft_cldice_iterations is None
                else soft_cldice_iterations
            ),
            smooth=self.cldice_smooth,
        )
        return {
            "geometry_aware_ce": geometry_ce,
            "dice": dice,
            "loci_cldice": loci_cldice,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        geometry_weights: torch.Tensor | None = None,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = self.components(
            logits, targets, geometry_weights, soft_cldice_iterations
        )
        return (
            self.geometry_aware_ce_weight * parts["geometry_aware_ce"]
            + self.dice_weight * parts["dice"]
            + self.soft_cldice_weight * parts["loci_cldice"]
        )
