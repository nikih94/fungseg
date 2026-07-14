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
    return float(((true_positives + smooth) / (predicted_positives + smooth)).mean().item())


def recall_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)
    true_positives = (predictions * targets).sum(dim=1)
    target_positives = targets.sum(dim=1)
    return float(((true_positives + smooth) / (target_positives + smooth)).mean().item())


def cldice_score_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 3,
    smooth: float = 1.0,
) -> float:
    import torch.nn.functional as F

    def as_nchw(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0).float()
        if tensor.ndim == 3:
            return tensor.unsqueeze(1).float()
        return tensor.float()

    def erode(mask: torch.Tensor) -> torch.Tensor:
        eroded_y = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
        eroded_x = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(eroded_x, eroded_y)

    def skeletonize(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0.0, 1.0)
        skeleton = F.relu(mask - F.max_pool2d(erode(mask), kernel_size=3, stride=1, padding=1))
        for _ in range(max(0, iterations - 1)):
            mask = erode(mask)
            delta = F.relu(mask - F.max_pool2d(erode(mask), kernel_size=3, stride=1, padding=1))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    predictions = as_nchw(predictions)
    targets = as_nchw(targets)
    prediction_skeleton = _flatten_batch(skeletonize(predictions))
    target_skeleton = _flatten_batch(skeletonize(targets))
    predictions = _flatten_batch(predictions)
    targets = _flatten_batch(targets)
    topology_precision = ((prediction_skeleton * targets).sum(dim=1) + smooth) / (
        prediction_skeleton.sum(dim=1) + smooth
    )
    topology_recall = ((target_skeleton * predictions).sum(dim=1) + smooth) / (
        target_skeleton.sum(dim=1) + smooth
    )
    cldice = (2.0 * topology_precision * topology_recall + smooth) / (
        topology_precision + topology_recall + smooth
    )
    return float(cldice.mean().item())


def multiclass_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).argmax(dim=1)


def multiclass_metrics_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: dict[str, int] | None = None,
    smooth: float = 1e-6,
    cldice_iterations: int = 3,
) -> dict[str, float]:
    """Per-class and foreground-macro metrics for class-index masks."""
    classes = class_names or {"loci": 1, "inoculum": 2}
    metrics: dict[str, float] = {}
    dice_values: list[float] = []
    iou_values: list[float] = []
    for name, class_id in classes.items():
        pred = (predictions == class_id).float()
        target = (targets == class_id).float()
        dice = dice_score_from_masks(pred, target, smooth=smooth)
        iou = iou_score_from_masks(pred, target, smooth=smooth)
        metrics[f"dice_{name}"] = dice
        metrics[f"iou_{name}"] = iou
        metrics[f"precision_{name}"] = precision_score_from_masks(pred, target, smooth=smooth)
        metrics[f"recall_{name}"] = recall_score_from_masks(pred, target, smooth=smooth)
        dice_values.append(dice)
        iou_values.append(iou)
    metrics["dice_macro_foreground"] = sum(dice_values) / max(len(dice_values), 1)
    metrics["iou_macro_foreground"] = sum(iou_values) / max(len(iou_values), 1)
    loci_id = classes.get("loci", 1)
    metrics["cldice_loci"] = cldice_score_from_masks(
        (predictions == loci_id).float(), (targets == loci_id).float(),
        iterations=cldice_iterations,
    )
    return metrics
