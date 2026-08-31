from __future__ import annotations

import argparse
from pathlib import Path
import warnings
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from src.data.discovery import (
    discover_image_mask_pairs,
    discover_image_mask_sets,
    discovery_diagnostic_messages,
)
from src.data.dataset import compose_multiclass_mask
from src.data.folds import make_csv_train_val_test_split
from src.inference.core import predict_probabilities_on_image, probabilities_to_binary_mask, resolve_device, save_rgb_image
from src.metrics.segmentation import (
    cldice_score_from_masks,
    dice_score_from_masks,
    iou_score_from_masks,
    join_region_metrics_from_masks,
    precision_score_from_masks,
    recall_score_from_masks,
    multiclass_metrics_from_masks,
)
from src.models.factory import build_model
from src.patching import OriginalImageRecord, build_original_image_records
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_mask_dir
from src.utils.io import ensure_dir, save_csv, save_json, save_mask_image

Predictor = Callable[[torch.nn.Module, Path, dict, torch.device], np.ndarray]


def _test_evaluation_config(config: dict) -> dict:
    """Return a config copy using 50% patch overlap for test prediction."""
    patching_config = dict(config["patching"])
    patch_size = int(patching_config["patch_size"])
    if patch_size < 2 or patch_size % 2:
        raise ValueError(
            "Test evaluation requires an even patching.patch_size of at least 2 "
            "to use a 50% stride."
        )
    patching_config["stride"] = patch_size // 2
    evaluation_config = dict(config)
    evaluation_config["patching"] = patching_config
    return evaluation_config


_BINARY_OVERLAY_COLORS = {
    "Ground truth only": np.array([50, 130, 255], dtype=np.uint8),
    "Prediction only": np.array([255, 135, 30], dtype=np.uint8),
    "Correct overlap": np.array([35, 220, 85], dtype=np.uint8),
}
_MULTICLASS_OVERLAY_COLORS = {
    "Loci ground truth only": np.array([50, 130, 255], dtype=np.uint8),
    "Loci prediction only": np.array([255, 135, 30], dtype=np.uint8),
    "Loci correct overlap": np.array([35, 220, 85], dtype=np.uint8),
    "Inoculum ground truth only": np.array([150, 95, 245], dtype=np.uint8),
    "Inoculum prediction only": np.array([245, 65, 155], dtype=np.uint8),
    "Inoculum correct overlap": np.array([30, 210, 220], dtype=np.uint8),
    "Wrong-class overlap": np.array([255, 210, 35], dtype=np.uint8),
}
_JOIN_MASK_BOUNDARY_COLOR = np.array([255, 0, 0], dtype=np.uint8)
_OVERLAY_ALPHA = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on the CSV test split.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--config", default=None, help="Configuration YAML. Defaults to the checkpoint run config.")
    parser.add_argument("--output", default=None, help="Directory for evaluation artifacts.")
    return parser.parse_args()


def default_config_path(checkpoint_path: Path) -> Path | None:
    path = checkpoint_path.parent.parent / "config.yaml"
    return path if path.is_file() else None


def resolve_test_records(config: dict) -> list[OriginalImageRecord]:
    split_cfg = config.get("split", {})
    split_mode = str(split_cfg.get("mode", "csv")).lower()
    if split_mode not in {"csv", "csv_kfold"}:
        raise ValueError(
            "Test evaluation requires split.mode: csv or csv_kfold."
        )
    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    if multiclass:
        pairs, diagnostics = discover_image_mask_sets(
            config["paths"]["images_dir"],
            {
                "loci": config["paths"]["mask_dirs"]["loci"],
                "inoculum": config["paths"]["mask_dirs"]["inoculum"],
            },
            config["data"]["image_extensions"],
            optional_mask_dirs=(
                {"join": config["join_masks"]["masks_dir"]}
                if (
                    config.get("join_masks", {}).get("enabled", False)
                    or config.get("join_masks", {}).get(
                        "evaluation_enabled", False
                    )
                )
                else None
            ),
        )
    else:
        pairs, diagnostics = discover_image_mask_pairs(
            config["paths"]["images_dir"], resolve_mask_dir(config), config["data"]["image_extensions"]
        )
    diagnostic_messages = discovery_diagnostic_messages(diagnostics)
    if diagnostic_messages:
        warnings.warn(
            "Test evaluation excluded incomplete or invalid image/mask sets: "
            + "; ".join(diagnostic_messages),
            RuntimeWarning,
            stacklevel=2,
        )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found for test evaluation.")
    records = build_original_image_records(pairs)
    split = make_csv_train_val_test_split(
        [record.source_id for record in records], split_cfg.get("csv_path", "data/image_splits.csv")
    )
    test_sources = set(split.test_sources)
    return [record for record in records if record.source_id in test_sources]


def threshold_values(config: dict) -> list[float]:
    evaluation_cfg = config.get("test_evaluation", {})
    start = float(evaluation_cfg.get("threshold_start", 0.5))
    stop = float(evaluation_cfg.get("threshold_stop", 1.0))
    step = float(evaluation_cfg.get("threshold_step", 0.01))
    if not (0.0 <= start <= stop <= 1.0) or step <= 0:
        raise ValueError("Test-evaluation thresholds must satisfy 0 <= start <= stop <= 1 and step > 0.")
    return [round(start + index * step, 10) for index in range(int(round((stop - start) / step)) + 1)]


def _tint_pixels(overlay: np.ndarray, pixels: np.ndarray, color: np.ndarray) -> None:
    overlay[pixels] = (
        (1.0 - _OVERLAY_ALPHA) * overlay[pixels] + _OVERLAY_ALPHA * color
    ).astype(np.uint8)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the inner four-connected boundary of a two-dimensional mask."""
    foreground = mask.astype(bool)
    interior = foreground.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    interior[1:-1, 1:-1] &= (
        foreground[:-2, 1:-1]
        & foreground[2:, 1:-1]
        & foreground[1:-1, :-2]
        & foreground[1:-1, 2:]
    )
    return foreground & ~interior


def _draw_overlay_legend(
    overlay: np.ndarray,
    legend_items: dict[str, np.ndarray],
) -> np.ndarray:
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    padding = 6
    swatch_size = 10
    row_height = 15
    text_width = max(draw.textbbox((0, 0), label, font=font)[2] for label in legend_items)
    box_width = padding * 3 + swatch_size + text_width
    box_height = padding * 2 + row_height * len(legend_items)
    left = max(0, image.width - box_width - 4)
    top = max(0, image.height - box_height - 4)
    right = image.width - 4
    bottom = image.height - 4
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=4,
        fill=(0, 0, 0, 190),
        outline=(255, 255, 255, 210),
    )
    for index, (label, color) in enumerate(legend_items.items()):
        row_top = top + padding + index * row_height
        draw.rectangle(
            (
                left + padding,
                row_top + 1,
                left + padding + swatch_size,
                row_top + swatch_size + 1,
            ),
            fill=(*map(int, color), 255),
            outline=(255, 255, 255, 230),
        )
        draw.text(
            (left + padding * 2 + swatch_size, row_top),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )
    return np.asarray(image, dtype=np.uint8)


def create_test_evaluation_overlay(
    original: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
    multiclass: bool,
    *,
    join_mask: np.ndarray | None = None,
    include_legend: bool = True,
) -> np.ndarray:
    """Overlay test ground truth and predictions with explicit error categories."""
    if original.shape[:2] != target_mask.shape or target_mask.shape != prediction_mask.shape:
        raise ValueError(
            "Image, ground-truth, and prediction shapes must match for an evaluation overlay."
        )
    if join_mask is not None and join_mask.shape != target_mask.shape:
        raise ValueError(
            "Join-mask and ground-truth shapes must match for an evaluation overlay."
        )
    overlay = np.array(original, dtype=np.uint8, copy=True)
    legend_items = dict(
        _MULTICLASS_OVERLAY_COLORS if multiclass else _BINARY_OVERLAY_COLORS
    )
    if multiclass:
        target = target_mask.astype(np.uint8)
        prediction = prediction_mask.astype(np.uint8)
        for class_id, class_name in ((1, "Loci"), (2, "Inoculum")):
            target_class = target == class_id
            prediction_class = prediction == class_id
            _tint_pixels(
                overlay,
                target_class & (prediction == 0),
                legend_items[f"{class_name} ground truth only"],
            )
            _tint_pixels(
                overlay,
                prediction_class & (target == 0),
                legend_items[f"{class_name} prediction only"],
            )
            _tint_pixels(
                overlay,
                target_class & prediction_class,
                legend_items[f"{class_name} correct overlap"],
            )
        _tint_pixels(
            overlay,
            (target > 0) & (prediction > 0) & (target != prediction),
            legend_items["Wrong-class overlap"],
        )
    else:
        target = target_mask.astype(bool)
        prediction = prediction_mask > 0
        _tint_pixels(overlay, target & ~prediction, legend_items["Ground truth only"])
        _tint_pixels(overlay, prediction & ~target, legend_items["Prediction only"])
        _tint_pixels(overlay, target & prediction, legend_items["Correct overlap"])
    if join_mask is not None:
        legend_items["Join-mask boundary"] = _JOIN_MASK_BOUNDARY_COLOR
        _tint_pixels(overlay, _mask_boundary(join_mask), _JOIN_MASK_BOUNDARY_COLOR)
    return _draw_overlay_legend(overlay, legend_items) if include_legend else overlay


def _metrics(
    probabilities: np.ndarray,
    target_mask: np.ndarray,
    threshold: float,
    metric_device: torch.device,
) -> dict[str, float]:
    prediction = torch.from_numpy((probabilities >= threshold).astype(np.float32)).to(metric_device)
    target = torch.from_numpy(target_mask.astype(np.float32)).to(metric_device)
    return {
        "dice": dice_score_from_masks(prediction, target),
        "iou": iou_score_from_masks(prediction, target),
        "precision": precision_score_from_masks(prediction, target),
        "recall": recall_score_from_masks(prediction, target),
        "cldice": cldice_score_from_masks(prediction, target),
        "predicted_foreground_fraction": float(prediction.mean().item()),
    }



def _multiclass_metrics(
    probabilities: np.ndarray,
    target_mask: np.ndarray,
    metric_device: torch.device,
) -> dict[str, float]:
    prediction = torch.from_numpy(probabilities.argmax(axis=0).astype(np.int64)).to(metric_device)
    target = torch.from_numpy(target_mask.astype(np.int64)).to(metric_device)
    return multiclass_metrics_from_masks(
        prediction, target, {"loci": 1, "inoculum": 2}
    )

def _save_threshold_plot(rows: list[dict], metric_name: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for source_id in sorted({str(row["source_id"]) for row in rows}):
        image_rows = [row for row in rows if row["source_id"] == source_id]
        axis.plot([row["threshold"] for row in image_rows], [row[metric_name] for row in image_rows], label=source_id)
    metric_labels = {
        "dice": "Dice/F1",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall",
        "cldice": "clDice",
        "predicted_foreground_fraction": "Predicted foreground fraction",
    }
    axis.set(xlabel="Foreground threshold", ylabel=metric_labels[metric_name], xlim=(0.5, 1.0), ylim=(0.0, 1.0))
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_test_evaluation(
    checkpoint_path: str | Path,
    config: dict,
    output_dir: str | Path,
    device: torch.device,
    model: torch.nn.Module | None = None,
    predictor: Predictor = predict_probabilities_on_image,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    output_dir = ensure_dir(output_dir)
    records = resolve_test_records(config)
    if not records:
        raise RuntimeError("The configured CSV split contains no test images.")
    if model is None:
        model = build_model(config["model"]).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)

    evaluation_config = _test_evaluation_config(config)
    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    inference_threshold = float(config.get("inference", {}).get("threshold", 0.5))
    mask_threshold = int(config["patching"]["mask_threshold"])
    thresholds = [] if multiclass else threshold_values(config)
    masks_dir, overlays_dir = ensure_dir(output_dir / "masks"), ensure_dir(output_dir / "overlays")
    probabilities_dir = output_dir / "probabilities"
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    metric_device = device if device.type == "cuda" else torch.device("cpu")
    for record in tqdm(records, desc="Test evaluation"):
        probabilities = predictor(model, record.image_path, evaluation_config, device)
        with Image.open(record.image_path) as image:
            image_array = np.array(image.convert("RGB"))
        if multiclass:
            if not record.mask_paths:
                raise ValueError("Multiclass test records require named masks.")
            with Image.open(record.mask_paths["loci"]) as mask:
                loci_mask = np.array(mask.convert("L"), dtype=np.uint8)
            with Image.open(record.mask_paths["inoculum"]) as mask:
                inoculum_mask = np.array(mask.convert("L"), dtype=np.uint8)
            join_mask = None
            if "join" in record.mask_paths:
                with Image.open(record.mask_paths["join"]) as mask:
                    join_mask = np.array(mask.convert("L"), dtype=np.uint8)
            join_masks_config = config.get("join_masks", {})
            merge_join_masks = bool(
                join_masks_config.get("enabled", False)
                and join_masks_config.get("merge_with_loci", False)
            )
            target_mask, overlap = compose_multiclass_mask(
                loci_mask,
                inoculum_mask,
                mask_threshold,
                join_mask=join_mask,
                merge_join_masks=merge_join_masks,
            )
            join_target_mask = target_mask
            if join_mask is not None and not merge_join_masks:
                join_target_mask, _ = compose_multiclass_mask(
                    loci_mask,
                    inoculum_mask,
                    mask_threshold,
                    join_mask=join_mask,
                    merge_join_masks=True,
                )
            if probabilities.shape != (3, *target_mask.shape):
                raise ValueError(f"Prediction shape does not match multiclass ground truth for {record.source_id}.")
            output_mask = probabilities.argmax(axis=0).astype(np.uint8)
            metrics = _multiclass_metrics(probabilities, target_mask, metric_device)
            metrics.update(overlap)
            metrics.update(join_region_metrics_from_masks(
                torch.from_numpy(output_mask).to(metric_device),
                torch.from_numpy(join_target_mask).to(metric_device),
                None if join_mask is None else torch.from_numpy(join_mask > mask_threshold).to(metric_device),
            ))
            if config.get("inference", {}).get("save_probabilities", False):
                save_mask_image(probabilities_dir / f"{record.image_path.stem}_prob_loci.png", probabilities[1] * 255.0)
                save_mask_image(probabilities_dir / f"{record.image_path.stem}_prob_inoculum.png", probabilities[2] * 255.0)
            mask_path_value = ",".join(
                str(record.mask_paths[name])
                for name in ("loci", "inoculum", "join")
                if name in record.mask_paths
            )
            threshold_value = "argmax"
        else:
            with Image.open(record.mask_path) as mask:
                target_mask = np.array(mask.convert("L"), dtype=np.uint8) > mask_threshold
            if probabilities.shape != target_mask.shape:
                raise ValueError(f"Prediction shape does not match ground truth for {record.source_id}.")
            output_mask = probabilities_to_binary_mask(probabilities, inference_threshold)
            metrics = _metrics(probabilities, target_mask, inference_threshold, metric_device)
            overlap = {}
            mask_path_value = str(record.mask_path)
            threshold_value = inference_threshold
        save_mask_image(masks_dir / f"{record.image_path.stem}_mask.png", output_mask)
        save_rgb_image(
            overlays_dir / f"{record.image_path.stem}_overlay.png",
            create_test_evaluation_overlay(
                image_array,
                target_mask,
                output_mask,
                multiclass,
                join_mask=(
                    None
                    if not multiclass or join_mask is None
                    else join_mask > mask_threshold
                ),
            ),
        )
        metric_rows.append({
            "source_id": record.source_id, "image_path": str(record.image_path), "mask_path": mask_path_value,
            "threshold": threshold_value, **metrics,
        })
        for threshold in tqdm(thresholds, desc=f"Thresholds | {record.source_id}", leave=False):
            threshold_rows.append({
                "source_id": record.source_id,
                "threshold": threshold,
                **_metrics(probabilities, target_mask, threshold, metric_device),
            })

    if multiclass:
        metric_names = (
            "dice_loci", "iou_loci", "precision_loci", "recall_loci",
            "dice_inoculum", "iou_inoculum", "precision_inoculum", "recall_inoculum",
            "dice_macro_foreground", "iou_macro_foreground", "cldice_loci",
            "overlap_pixels", "overlap_fraction",
            "join_pixels", "dice_join", "iou_join",
        )
    else:
        metric_names = ("dice", "iou", "precision", "recall", "cldice", "predicted_foreground_fraction")
    metric_rows.append({
        "source_id": "mean", "image_path": "", "mask_path": "",
        "threshold": "argmax" if multiclass else inference_threshold,
        **{
            metric_name: (
                float(np.mean([row[metric_name] for row in metric_rows if row[metric_name] is not None]))
                if any(row[metric_name] is not None for row in metric_rows)
                else None
            )
            for metric_name in metric_names
        },
    })
    save_csv(output_dir / "test_metrics.csv", metric_rows)
    save_csv(output_dir / "threshold_metrics.csv", threshold_rows)
    if not multiclass:
        for metric_name in metric_names:
            _save_threshold_plot(threshold_rows, metric_name, output_dir / f"{metric_name}_by_threshold.png")
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        plotted = ["dice_loci", "dice_inoculum", "dice_macro_foreground", "cldice_loci"]
        axis.bar(plotted, [metric_rows[-1][name] for name in plotted])
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(output_dir / "multiclass_metrics.png", dpi=160)
        plt.close(figure)
    result = {
        "checkpoint": str(checkpoint_path), "output_dir": str(output_dir), "num_test_images": len(records),
        "threshold": "argmax" if multiclass else inference_threshold,
        **{f"mean_{metric_name}": metric_rows[-1][metric_name] for metric_name in metric_names},
    }
    if multiclass:
        result["num_join_images"] = sum(
            row.get("dice_join") is not None for row in metric_rows[:-1]
        )
        result["mean_dice"] = result["mean_dice_macro_foreground"]
        result["mean_iou"] = result["mean_iou_macro_foreground"]
    save_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else default_config_path(checkpoint_path)
    if config_path is None:
        raise ValueError("No run config was found next to the checkpoint. Pass --config explicitly.")
    config = load_config(config_path)
    device = resolve_device(str(config["train"].get("device", "auto")))
    output_dir = Path(args.output) if args.output else checkpoint_path.parent.parent / "test-evaluation"
    result = run_test_evaluation(checkpoint_path, config, output_dir, device)
    print(f"Test evaluation complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
