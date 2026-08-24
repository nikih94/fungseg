from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.data.dataset import compose_multiclass_mask
from src.data.discovery import (
    discover_image_mask_pairs,
    discover_image_mask_sets,
    discovery_diagnostic_messages,
)
from src.data.folds import (
    make_csv_train_val_test_split,
    make_grouped_kfold_splits,
    make_manual_train_val_split,
)
from src.inference.core import (
    predict_probabilities_on_image,
    probabilities_to_binary_mask,
    resolve_device,
    save_rgb_image,
)
from src.inference.test_evaluation import (
    _metrics,
    _multiclass_metrics,
    create_test_evaluation_overlay,
)
from src.metrics.segmentation import join_region_metrics_from_masks
from src.models.factory import build_model
from src.patching import OriginalImageRecord, build_original_image_records
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_mask_dir
from src.utils.io import ensure_dir, save_csv, save_json

Predictor = Callable[[torch.nn.Module, Path, dict, torch.device], np.ndarray]
_FOLD_PATTERN = re.compile(r"^fold_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation checkpoint on its train and validation image sets."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the best checkpoint to evaluate.")
    parser.add_argument("--config", default=None, help="Configuration YAML; defaults to the checkpoint run config.")
    parser.add_argument("--output", default=None, help="Output directory; defaults to the checkpoint run directory.")
    return parser.parse_args()


def default_config_path(checkpoint_path: Path) -> Path | None:
    path = checkpoint_path.parent.parent / "config.yaml"
    return path if path.is_file() else None


def _discover_records(config: dict) -> list[OriginalImageRecord]:
    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    if multiclass:
        pairs, diagnostics = discover_image_mask_sets(
            config["paths"]["images_dir"],
            {"loci": config["paths"]["mask_dirs"]["loci"], "inoculum": config["paths"]["mask_dirs"]["inoculum"]},
            config["data"]["image_extensions"],
            optional_mask_dirs=(
                {"join": config["join_masks"]["masks_dir"]}
                if config.get("join_masks", {}).get("enabled", False)
                else None
            ),
        )
    else:
        pairs, diagnostics = discover_image_mask_pairs(
            config["paths"]["images_dir"], resolve_mask_dir(config), config["data"]["image_extensions"]
        )
    messages = discovery_diagnostic_messages(diagnostics)
    if messages:
        warnings.warn(
            "Train/validation evaluation excluded incomplete or invalid image/mask sets: "
            + "; ".join(messages),
            RuntimeWarning,
            stacklevel=2,
        )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found for train/validation evaluation.")
    return build_original_image_records(pairs)


def resolve_train_validation_records(
    config: dict, checkpoint_path: str | Path | None = None
) -> tuple[list[OriginalImageRecord], list[OriginalImageRecord]]:
    records = _discover_records(config)
    source_ids = [record.source_id for record in records]
    split_cfg = config.get("split", {})
    mode = str(split_cfg.get("mode", "csv")).lower()
    if mode == "csv":
        split = make_csv_train_val_test_split(source_ids, split_cfg.get("csv_path", "data/image_splits.csv"))
        train_sources, val_sources = split.train_sources, split.val_sources
    elif mode == "train_val":
        train_sources, val_sources = make_manual_train_val_split(
            source_ids, split_cfg.get("val_source_ids", [])
        )[0]
    elif mode == "kfold":
        if checkpoint_path is None:
            raise ValueError("A checkpoint path is required to resolve a k-fold train/validation split.")
        match = _FOLD_PATTERN.match(Path(checkpoint_path).parent.name)
        if match is None:
            raise ValueError("A k-fold checkpoint must be stored under a fold_<n> directory.")
        folds = make_grouped_kfold_splits(
            source_ids,
            int(split_cfg.get("n_splits", 5)),
            bool(split_cfg.get("shuffle", True)),
            split_cfg.get("random_state"),
        )
        fold_index = int(match.group(1))
        if fold_index >= len(folds):
            raise ValueError(f"Checkpoint fold index {fold_index} is outside the configured k-fold split.")
        train_sources, val_sources = folds[fold_index]
    else:
        raise ValueError(f"Unsupported split mode '{mode}'. Expected csv, train_val, or kfold.")
    train_set, val_set = set(train_sources), set(val_sources)
    return (
        [record for record in records if record.source_id in train_set],
        [record for record in records if record.source_id in val_set],
    )


def _evaluate_records(
    records: list[OriginalImageRecord],
    split_name: str,
    model: torch.nn.Module,
    config: dict,
    output_dir: Path,
    device: torch.device,
    predictor: Predictor,
) -> tuple[list[dict], tuple[str, ...]]:
    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    mask_threshold = int(config["patching"]["mask_threshold"])
    metric_device = device if device.type == "cuda" else torch.device("cpu")
    overlays_dir = ensure_dir(output_dir / "overlays" / split_name)
    rows: list[dict] = []
    metric_names = (
        ("dice_loci", "iou_loci", "precision_loci", "recall_loci", "dice_inoculum", "iou_inoculum",
         "precision_inoculum", "recall_inoculum", "dice_macro_foreground", "iou_macro_foreground",
         "cldice_loci", "overlap_pixels", "overlap_fraction",
         "join_pixels", "dice_join", "iou_join")
        if multiclass else
        ("dice", "iou", "precision", "recall", "cldice", "predicted_foreground_fraction")
    )
    for record in tqdm(records, desc=f"{split_name.title()} evaluation"):
        probabilities = predictor(model, record.image_path, config, device)
        with Image.open(record.image_path) as image:
            image_array = np.array(image.convert("RGB"))
        if multiclass:
            if not record.mask_paths:
                raise ValueError("Multiclass records require named masks.")
            with Image.open(record.mask_paths["loci"]) as mask:
                loci = np.array(mask.convert("L"), dtype=np.uint8)
            with Image.open(record.mask_paths["inoculum"]) as mask:
                inoculum = np.array(mask.convert("L"), dtype=np.uint8)
            join_mask = None
            if "join" in record.mask_paths:
                with Image.open(record.mask_paths["join"]) as mask:
                    join_mask = np.array(mask.convert("L"), dtype=np.uint8)
            target, overlap = compose_multiclass_mask(
                loci,
                inoculum,
                mask_threshold,
                join_mask=join_mask,
                merge_join_masks=bool(
                    config.get("join_masks", {}).get("merge_with_loci", False)
                ),
            )
            if probabilities.shape != (3, *target.shape):
                raise ValueError(f"Prediction shape does not match ground truth for {record.source_id}.")
            prediction = probabilities.argmax(axis=0).astype(np.uint8)
            metrics = _multiclass_metrics(probabilities, target, metric_device)
            metrics.update(overlap)
            metrics.update(join_region_metrics_from_masks(
                torch.from_numpy(prediction).to(metric_device),
                torch.from_numpy(target).to(metric_device),
                None if join_mask is None else torch.from_numpy(join_mask > mask_threshold).to(metric_device),
            ))
            threshold_value = "argmax"
            mask_path = ",".join(
                str(record.mask_paths[name])
                for name in ("loci", "inoculum", "join")
                if name in record.mask_paths
            )
        else:
            with Image.open(record.mask_path) as mask:
                target = np.array(mask.convert("L"), dtype=np.uint8) > mask_threshold
            if probabilities.shape != target.shape:
                raise ValueError(f"Prediction shape does not match ground truth for {record.source_id}.")
            prediction = probabilities_to_binary_mask(probabilities, threshold)
            metrics = _metrics(probabilities, target, threshold, metric_device)
            threshold_value, mask_path = threshold, str(record.mask_path)
        save_rgb_image(
            overlays_dir / f"{record.image_path.stem}_overlay.png",
            create_test_evaluation_overlay(image_array, target, prediction, multiclass),
        )
        rows.append({"split": split_name, "source_id": record.source_id, "image_path": str(record.image_path),
                     "mask_path": mask_path, "threshold": threshold_value, **metrics})
    return rows, metric_names


def _mean_row(rows: list[dict], split: str, metric_names: tuple[str, ...]) -> dict:
    return {
        "split": split, "source_id": "mean", "image_path": "", "mask_path": "",
        "threshold": rows[0]["threshold"],
        **{
            name: (
                float(np.mean([row[name] for row in rows if row[name] is not None]))
                if any(row[name] is not None for row in rows)
                else None
            )
            for name in metric_names
        },
    }


def run_val_train_set_evaluation(
    checkpoint_path: str | Path, config: dict, output_dir: str | Path, device: torch.device,
    model: torch.nn.Module | None = None, predictor: Predictor = predict_probabilities_on_image,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    output_dir = ensure_dir(output_dir)
    train_records, val_records = resolve_train_validation_records(config, checkpoint_path)
    if not train_records or not val_records:
        raise RuntimeError("Both train and validation sets must contain at least one image.")
    if model is None:
        model = build_model(config["model"]).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
    train_rows, metric_names = _evaluate_records(train_records, "train", model, config, output_dir, device, predictor)
    val_rows, _ = _evaluate_records(val_records, "validation", model, config, output_dir, device, predictor)
    train_mean = _mean_row(train_rows, "train_mean", metric_names)
    val_mean = _mean_row(val_rows, "validation_mean", metric_names)
    combined = _mean_row(train_rows + val_rows, "train_validation_mean", metric_names)
    rows = train_rows + val_rows + [train_mean, val_mean, combined]
    save_csv(output_dir / "val_train_set_metrics.csv", rows)
    result = {
        "checkpoint": str(checkpoint_path), "output_dir": str(output_dir),
        "num_train_images": len(train_records), "num_validation_images": len(val_records),
        **{f"train_mean_{name}": train_mean[name] for name in metric_names},
        **{f"validation_mean_{name}": val_mean[name] for name in metric_names},
        **{f"train_validation_mean_{name}": combined[name] for name in metric_names},
    }
    if "dice_macro_foreground" in metric_names:
        result["train_mean_dice"] = train_mean["dice_macro_foreground"]
        result["validation_mean_dice"] = val_mean["dice_macro_foreground"]
        result["train_validation_mean_dice"] = combined["dice_macro_foreground"]
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
    output_dir = Path(args.output) if args.output else checkpoint_path.parent.parent / "val-train-set-evaluation"
    result = run_val_train_set_evaluation(checkpoint_path, config, output_dir, device)
    print(f"Train/validation evaluation complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
