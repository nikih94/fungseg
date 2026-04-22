from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).float()


def _sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).float()


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


def _soft_cldice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    iterations: int,
    smooth: float,
) -> torch.Tensor:
    predictions = predictions.float()
    targets = targets.float()
    prediction_skeleton = _soft_skeletonize(predictions, iterations)
    target_skeleton = _soft_skeletonize(targets, iterations)

    prediction_skeleton = _flatten_batch(prediction_skeleton)
    target_skeleton = _flatten_batch(target_skeleton)
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)

    topology_precision = ((prediction_skeleton * targets).sum(dim=1) + smooth) / (
        prediction_skeleton.sum(dim=1) + smooth
    )
    topology_sensitivity = ((target_skeleton * predictions).sum(dim=1) + smooth) / (
        target_skeleton.sum(dim=1) + smooth
    )
    return (2.0 * topology_precision * topology_sensitivity + smooth) / (
        topology_precision + topology_sensitivity + smooth
    )


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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_dice_loss = self.bce_dice(logits, targets)
        soft_cldice_loss = self.soft_cldice(logits, targets)
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
    def __init__(self, threshold: float = 0.5, iterations: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = _sigmoid_logits(logits)
        hard_predictions = (probabilities >= self.threshold).float()
        cldice_score = _soft_cldice_score(
            hard_predictions,
            targets,
            iterations=self.iterations,
            smooth=self.smooth,
        )
        return 1.0 - cldice_score.mean()


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = _sigmoid_logits(logits)
        cldice_score = _soft_cldice_score(
            probabilities,
            targets,
            iterations=self.iterations,
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        soft_cldice_loss = self.soft_cldice(logits, targets)
        return (self.tversky_weight * tversky_loss) + (self.soft_cldice_weight * soft_cldice_loss)
