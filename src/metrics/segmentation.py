from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize


# PyTorch core does not currently ship Dice/IoU segmentation metrics,
# so these remain lightweight project-local implementations.
def _prepare_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).float()


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(tensor.shape[0], -1).float()


def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0).float()
    if tensor.ndim == 3:
        return tensor.unsqueeze(1).float()
    if tensor.ndim == 4:
        return tensor.float()
    raise ValueError(f"Expected a 2D, 3D, or 4D mask tensor, got shape {tuple(tensor.shape)}.")


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    eroded_y = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    eroded_x = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_x, eroded_y)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    eroded = _soft_erode(mask)
    return F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Return the differentiable morphological skeleton used by soft-clDice."""
    mask = mask.float().clamp(0.0, 1.0)
    skeleton = F.relu(mask - _soft_open(mask))
    for _ in range(max(0, iterations)):
        mask = _soft_erode(mask)
        delta = F.relu(mask - _soft_open(mask))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def _cldice_from_skeletons(
    prediction_skeleton: torch.Tensor,
    target_skeleton: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float,
) -> torch.Tensor:
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
    denominator = topology_precision + topology_sensitivity
    return torch.where(
        denominator > 0,
        2.0 * topology_precision * topology_sensitivity / denominator,
        torch.zeros_like(denominator),
    )


def soft_cldice_scores_from_probabilities(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    iterations: int | Sequence[int] | torch.Tensor = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return differentiable per-sample clDice scores for normalized predictions."""
    predictions = _as_nchw(predictions)
    targets = _as_nchw(targets)
    if not isinstance(iterations, int):
        values = (
            iterations.detach().cpu().reshape(-1).tolist()
            if isinstance(iterations, torch.Tensor)
            else list(iterations)
        )
        values = [int(value) for value in values]
        if len(values) != predictions.shape[0]:
            raise ValueError(
                "Per-sample Soft-clDice iterations must match the batch size: "
                f"iterations={len(values)}, batch={predictions.shape[0]}."
            )
        if any(value < 0 for value in values):
            raise ValueError("Soft-clDice iterations must be non-negative.")
        scores: list[torch.Tensor | None] = [None] * len(values)
        for value in sorted(set(values)):
            indices = [index for index, item in enumerate(values) if item == value]
            group_scores = soft_cldice_scores_from_probabilities(
                predictions[indices],
                targets[indices],
                iterations=value,
                smooth=smooth,
            )
            for index, score in zip(indices, group_scores):
                scores[index] = score
        return torch.stack([score for score in scores if score is not None])
    return _cldice_from_skeletons(
        soft_skeletonize(predictions, iterations),
        soft_skeletonize(targets, iterations),
        predictions,
        targets,
        smooth,
    )


def hard_skeletonize_masks(mask: torch.Tensor) -> torch.Tensor:
    """Return paper-reference Zhang skeletons for independent binary masks.

    scikit-image runs on CPU. The returned NCHW boolean tensor is moved back to
    the input device so callers that score GPU-resident patch masks remain
    device-compatible.
    """
    prepared = _as_nchw(mask)
    binary = prepared.detach().cpu().numpy() > 0.5
    skeletons = np.empty_like(binary, dtype=np.bool_)
    for batch_index in range(binary.shape[0]):
        for channel_index in range(binary.shape[1]):
            skeletons[batch_index, channel_index] = skeletonize(
                binary[batch_index, channel_index],
                method="zhang",
            )
    return torch.from_numpy(skeletons).to(device=prepared.device)


def cldice_score_from_skeletons(
    prediction_skeleton: torch.Tensor,
    target_skeleton: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Return hard clDice using caller-supplied skeletons."""
    score = _cldice_from_skeletons(
        prediction_skeleton,
        target_skeleton,
        _as_nchw(predictions),
        _as_nchw(targets),
        smooth,
    )
    return float(score.mean().item())


def dice_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    score = dice_scores_from_masks(predictions, targets, smooth=smooth)
    return float(score.mean().item())


def dice_scores_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)

    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    score = iou_scores_from_masks(predictions, targets, smooth=smooth)
    return float(score.mean().item())


def iou_scores_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    return dice_score_from_masks(predictions, targets, smooth=smooth)


def dice_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    predictions = _prepare_predictions(logits, threshold)
    return dice_scores_from_masks(predictions, targets, smooth=smooth)


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    predictions = _prepare_predictions(logits, threshold)
    return iou_score_from_masks(predictions, targets, smooth=smooth)


def iou_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    predictions = _prepare_predictions(logits, threshold)
    return iou_scores_from_masks(predictions, targets, smooth=smooth)


def precision_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)
    true_positives = (predictions * targets).sum(dim=1)
    predicted_positives = predictions.sum(dim=1)
    target_positives = targets.sum(dim=1)
    score = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives.clamp_min(smooth),
        (target_positives == 0).float(),
    )
    return float(score.mean().item())


def recall_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)
    true_positives = (predictions * targets).sum(dim=1)
    target_positives = targets.sum(dim=1)
    predicted_positives = predictions.sum(dim=1)
    score = torch.where(
        target_positives > 0,
        true_positives / target_positives.clamp_min(smooth),
        (predicted_positives == 0).float(),
    )
    return float(score.mean().item())


def cldice_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    *,
    prediction_skeleton: torch.Tensor | None = None,
    target_skeleton: torch.Tensor | None = None,
) -> float:
    predictions = _as_nchw(predictions)
    targets = _as_nchw(targets)
    if prediction_skeleton is None:
        prediction_skeleton = hard_skeletonize_masks(predictions)
    else:
        prediction_skeleton = _as_nchw(prediction_skeleton).to(predictions.device)
    if target_skeleton is None:
        target_skeleton = hard_skeletonize_masks(targets)
    else:
        target_skeleton = _as_nchw(target_skeleton).to(targets.device)
    return cldice_score_from_skeletons(
        prediction_skeleton,
        target_skeleton,
        predictions,
        targets,
        smooth,
    )


def multiclass_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).argmax(dim=1)


def multiclass_metrics_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: dict[str, int] | None = None,
    smooth: float = 1e-6,
    *,
    loci_target_skeleton: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return per-class and foreground-macro metrics for class-index masks."""
    classes = class_names or {"loci": 1, "inoculum": 2}
    metrics: dict[str, float] = {}
    dice_values: list[torch.Tensor] = []
    iou_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    for name, class_id in classes.items():
        pred = (predictions == class_id).float()
        target = (targets == class_id).float()
        dice = dice_scores_from_masks(pred, target, smooth=smooth)
        iou = iou_scores_from_masks(pred, target, smooth=smooth)
        valid = (_flatten_batch(pred).sum(dim=1) + _flatten_batch(target).sum(dim=1)) > 0
        metrics[f"dice_{name}"] = float(dice.mean().item())
        metrics[f"iou_{name}"] = float(iou.mean().item())
        metrics[f"precision_{name}"] = precision_score_from_masks(pred, target, smooth=smooth)
        metrics[f"recall_{name}"] = recall_score_from_masks(pred, target, smooth=smooth)
        dice_values.append(dice)
        iou_values.append(iou)
        valid_values.append(valid)

    def macro_present(scores: list[torch.Tensor], validity: list[torch.Tensor]) -> float:
        if not scores:
            return 1.0
        stacked_scores = torch.stack(scores, dim=1)
        stacked_validity = torch.stack(validity, dim=1)
        counts = stacked_validity.sum(dim=1)
        per_sample = (stacked_scores * stacked_validity).sum(dim=1) / counts.clamp_min(1)
        per_sample = torch.where(counts > 0, per_sample, torch.ones_like(per_sample))
        return float(per_sample.mean().item())

    metrics["dice_macro_foreground"] = macro_present(dice_values, valid_values)
    metrics["iou_macro_foreground"] = macro_present(iou_values, valid_values)
    loci_id = classes.get("loci", 1)
    metrics["cldice_loci"] = cldice_score_from_masks(
        (predictions == loci_id).float(),
        (targets == loci_id).float(),
        target_skeleton=loci_target_skeleton,
    )
    return metrics


def join_region_metrics_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    join_mask: torch.Tensor | None,
    loci_class_id: int = 1,
) -> dict[str, float | int | None]:
    """Score loci recovery inside annotated join regions, excluding absent regions."""
    if join_mask is None:
        return {"join_pixels": 0, "dice_join": None, "iou_join": None}
    effective_join = (join_mask > 0) & (targets == loci_class_id)
    join_pixels = int(effective_join.sum().item())
    if join_pixels == 0:
        return {"join_pixels": 0, "dice_join": None, "iou_join": None}
    prediction = ((predictions == loci_class_id) & effective_join).float()
    target = effective_join.float()
    return {
        "join_pixels": join_pixels,
        "dice_join": dice_score_from_masks(prediction, target),
        "iou_join": iou_score_from_masks(prediction, target),
    }
