from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from src.data.dataset import get_val_transforms
from src.data.discovery import discover_image_mask_pairs
from src.inference import create_overlay
from src.metrics.segmentation import dice_score_from_masks, iou_score_from_masks
from src.models.factory import build_model
from src.models.wrappers import extract_logits
from src.patching import _compute_positions, crop_and_pad_array
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_csv


@dataclass(frozen=True)
class CheckpointEntry:
    fold: int
    checkpoint: str
    path: Path
    reason: str
    epoch: int | None
    epoch_start: int | None
    epoch_end: int | None
    monitor: str
    monitor_value: float | None


@dataclass(frozen=True)
class SelectedCrop:
    x: int
    y: int
    width: int
    height: int
    foreground_ratio: float
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare run checkpoints on qualitative crops.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--config", default=None, help="Config path. Defaults to <run-dir>/config.yaml.")
    parser.add_argument("--data-root", default=None, help="Qualitative data root with images/ and masks/.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <run-dir>/qualitative_evaluation.")
    parser.add_argument("--crop-patch-grid", nargs=2, type=int, default=None, metavar=("ROWS", "COLS"))
    parser.add_argument("--min-foreground-ratio", type=float, default=None)
    parser.add_argument("--max-foreground-ratio", type=float, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _checkpoint_path(row: dict[str, str], fold_dir: Path) -> Path:
    raw_path = row.get("path") or row.get("checkpoint") or ""
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    run_dir = fold_dir.parent
    candidates = [run_dir / path, fold_dir / path]
    if run_dir.name in path.parts:
        run_name_index = path.parts.index(run_dir.name)
        suffix = Path(*path.parts[run_name_index + 1 :])
        candidates.append(run_dir / suffix)
    candidates.append(fold_dir / str(row.get("checkpoint", raw_path)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def discover_manifest_checkpoints(run_dir: str | Path, max_checkpoints: int | None = None) -> list[CheckpointEntry]:
    run_dir = Path(run_dir)
    entries: list[CheckpointEntry] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_index = int(fold_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        manifest_path = fold_dir / "checkpoint_manifest.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                checkpoint_name = row.get("checkpoint", "")
                if not checkpoint_name:
                    continue
                checkpoint_path = _checkpoint_path(row, fold_dir)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint listed in manifest does not exist: {checkpoint_path}")
                entries.append(
                    CheckpointEntry(
                        fold=fold_index,
                        checkpoint=checkpoint_name,
                        path=checkpoint_path,
                        reason=row.get("reason", ""),
                        epoch=_as_int(row.get("epoch")),
                        epoch_start=_as_int(row.get("epoch_start")),
                        epoch_end=_as_int(row.get("epoch_end")),
                        monitor=row.get("monitor", ""),
                        monitor_value=_as_float(row.get("monitor_value")),
                    )
                )
                if max_checkpoints is not None and len(entries) >= max_checkpoints:
                    return entries
    return entries


def select_qualitative_crop(
    mask_array: np.ndarray,
    patch_size: int,
    stride: int,
    crop_patch_grid: tuple[int, int],
    mask_threshold: int,
    min_foreground_ratio: float,
    max_foreground_ratio: float,
    rng: np.random.Generator | None = None,
) -> SelectedCrop:
    rows, cols = crop_patch_grid
    if rows <= 0 or cols <= 0:
        raise ValueError(f"crop_patch_grid must contain positive values, got {crop_patch_grid}.")

    height, width = mask_array.shape[:2]
    xs = _compute_positions(width, patch_size, stride)
    ys = _compute_positions(height, patch_size, stride)
    max_x_index = max(len(xs) - cols, 0)
    max_y_index = max(len(ys) - rows, 0)
    midpoint = (min_foreground_ratio + max_foreground_ratio) / 2.0
    in_range: list[tuple[float, int, int, SelectedCrop]] = []
    non_empty: list[tuple[float, int, int, SelectedCrop]] = []
    all_crops: list[tuple[float, int, int, SelectedCrop]] = []

    binary_mask = mask_array > mask_threshold
    for y_index in range(max_y_index + 1):
        for x_index in range(max_x_index + 1):
            x0 = xs[x_index]
            y0 = ys[y_index]
            x_last = xs[min(x_index + cols - 1, len(xs) - 1)]
            y_last = ys[min(y_index + rows - 1, len(ys) - 1)]
            x1 = min(width, x_last + patch_size)
            y1 = min(height, y_last + patch_size)
            crop_mask = binary_mask[y0:y1, x0:x1]
            foreground_ratio = float(crop_mask.mean()) if crop_mask.size else 0.0
            crop = SelectedCrop(
                x=x0,
                y=y0,
                width=max(0, x1 - x0),
                height=max(0, y1 - y0),
                foreground_ratio=foreground_ratio,
                selection_reason="in_range",
            )
            score = abs(foreground_ratio - midpoint)
            all_crops.append((score, y0, x0, crop))
            if min_foreground_ratio <= foreground_ratio <= max_foreground_ratio:
                in_range.append((score, y0, x0, crop))
            if foreground_ratio > 0:
                non_empty.append((foreground_ratio, y0, x0, crop))

    def choose_candidate(candidates: list[tuple[float, int, int, SelectedCrop]]) -> SelectedCrop:
        ordered = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))
        if rng is None:
            return ordered[0][3]
        return ordered[int(rng.integers(0, len(ordered)))][3]

    if in_range:
        return choose_candidate(in_range)
    if non_empty:
        crop = choose_candidate(non_empty)
        selection_reason = "lowest_non_empty" if rng is None else "non_empty"
        return replace(crop, selection_reason=selection_reason)
    crop = choose_candidate(all_crops)
    return replace(crop, selection_reason="no_foreground")


def intersecting_patch_coordinates(
    image_width: int,
    image_height: int,
    patch_size: int,
    stride: int,
    crop: SelectedCrop,
) -> list[tuple[int, int]]:
    crop_x1 = crop.x + crop.width
    crop_y1 = crop.y + crop.height
    coordinates: list[tuple[int, int]] = []
    for y in _compute_positions(image_height, patch_size, stride):
        patch_y1 = min(image_height, y + patch_size)
        if y >= crop_y1 or patch_y1 <= crop.y:
            continue
        for x in _compute_positions(image_width, patch_size, stride):
            patch_x1 = min(image_width, x + patch_size)
            if x >= crop_x1 or patch_x1 <= crop.x:
                continue
            coordinates.append((x, y))
    return coordinates


def predict_crop_probabilities(
    model,
    image_array: np.ndarray,
    crop: SelectedCrop,
    config: dict[str, Any],
    device: torch.device,
    use_amp: bool = False,
) -> np.ndarray:
    data_cfg = config["data"]
    patch_size = int(data_cfg["patch_size"])
    stride = int(data_cfg["stride"])
    transforms = get_val_transforms(
        data_cfg.get("image_size"),
        augmentations_config=config.get("augmentations", {}),
    )
    height, width = image_array.shape[:2]
    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, y in intersecting_patch_coordinates(width, height, patch_size, stride, crop):
            image_patch = crop_and_pad_array(image_array, x, y, patch_size)
            transformed = transforms(
                image=image_patch,
                mask=np.zeros((patch_size, patch_size), dtype=np.float32),
            )
            image_tensor = transformed["image"].unsqueeze(0).to(device, non_blocking=True)
            autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                logits = extract_logits(model(image_tensor))
                probabilities = torch.sigmoid(logits)
            if probabilities.shape[-2:] != (patch_size, patch_size):
                probabilities = F.interpolate(
                    probabilities,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
            probability_patch = probabilities.squeeze().cpu().numpy().astype(np.float32)
            valid_height = min(patch_size, height - y)
            valid_width = min(patch_size, width - x)
            probability_sum[y : y + valid_height, x : x + valid_width] += probability_patch[
                :valid_height, :valid_width
            ]
            probability_count[y : y + valid_height, x : x + valid_width] += 1.0

    averaged = probability_sum / np.clip(probability_count, a_min=1.0, a_max=None)
    return averaged[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]


def metric_row(
    image_path: Path,
    entry: CheckpointEntry,
    crop: SelectedCrop,
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    prediction_tensor = torch.from_numpy((prediction_mask > 0).astype(np.float32))
    target_tensor = torch.from_numpy((target_mask > 0).astype(np.float32))
    return {
        "image": image_path.name,
        "image_stem": image_path.stem,
        "fold": entry.fold,
        "checkpoint": entry.checkpoint,
        "checkpoint_path": str(entry.path),
        "reason": entry.reason,
        "epoch": entry.epoch,
        "epoch_start": entry.epoch_start,
        "epoch_end": entry.epoch_end,
        "monitor": entry.monitor,
        "monitor_value": entry.monitor_value,
        "crop_x": crop.x,
        "crop_y": crop.y,
        "crop_width": crop.width,
        "crop_height": crop.height,
        "crop_foreground_ratio": crop.foreground_ratio,
        "dice": dice_score_from_masks(prediction_tensor, target_tensor),
        "iou": iou_score_from_masks(prediction_tensor, target_tensor),
    }


def crop_row(image_path: Path, mask_path: Path, crop: SelectedCrop, selection_seed: int | None) -> dict[str, Any]:
    return {
        "image": image_path.name,
        "mask": mask_path.name,
        "image_stem": image_path.stem,
        "selection_seed": selection_seed,
        "crop_x": crop.x,
        "crop_y": crop.y,
        "crop_width": crop.width,
        "crop_height": crop.height,
        "foreground_ratio": crop.foreground_ratio,
        "selection_reason": crop.selection_reason,
    }


def _resize_to_panel(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_uint8, mode="L").convert("RGB")


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default()
    draw.multiline_text(xy, text, fill=(20, 20, 20), font=font, spacing=2)


def make_panel(title: str, base: np.ndarray, mask: np.ndarray | None = None, panel_size: int = 512) -> Image.Image:
    label_height = 46
    panel = Image.new("RGB", (panel_size, panel_size + label_height), "white")
    draw = ImageDraw.Draw(panel)
    _draw_label(draw, (8, 6), title[:120])

    content = _resize_to_panel(Image.fromarray(base.astype(np.uint8)), panel_size)
    panel.paste(content, (0, label_height))
    if mask is not None:
        inset_size = max(96, panel_size // 3)
        inset = _mask_to_rgb(mask).resize((inset_size, inset_size), resample=Image.Resampling.NEAREST)
        inset_x = panel_size - inset_size - 8
        inset_y = label_height + panel_size - inset_size - 8
        draw.rectangle(
            [inset_x - 2, inset_y - 2, inset_x + inset_size + 1, inset_y + inset_size + 1],
            fill=(255, 255, 255),
            outline=(30, 30, 30),
        )
        panel.paste(inset, (inset_x, inset_y))
    return panel


def save_grid(path: Path, panels: list[Image.Image], columns: int = 3) -> None:
    if not panels:
        return
    columns = max(1, columns)
    panel_width, panel_height = panels[0].size
    rows = int(math.ceil(len(panels) / columns))
    grid = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        grid.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def _checkpoint_label(entry: CheckpointEntry) -> str:
    monitor = "n/a" if entry.monitor_value is None else f"{entry.monitor_value:.4f}"
    epoch = "?" if entry.epoch is None else str(entry.epoch)
    return f"fold {entry.fold} | {entry.checkpoint}\nepoch {epoch} | {entry.monitor}={monitor}"


def run_qualitative_evaluation(
    run_dir: str | Path,
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    crop_patch_grid: tuple[int, int] | None = None,
    min_foreground_ratio: float | None = None,
    max_foreground_ratio: float | None = None,
    selection_seed: int | None = None,
    threshold: float | None = None,
    device_name: str | None = None,
    max_checkpoints: int | None = None,
    logger=None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = load_config(config_path or run_dir / "config.yaml")
    qualitative_cfg = config.get("qualitative_evaluation", {})
    data_root = Path(data_root or qualitative_cfg.get("data_root", "data/qualitative_evaluation"))
    output_dir = ensure_dir(output_dir or run_dir / "qualitative_evaluation")
    grids_dir = ensure_dir(output_dir / "grids")
    crop_patch_grid = tuple(crop_patch_grid or qualitative_cfg.get("crop_patch_grid", [3, 3]))
    min_foreground_ratio = float(
        qualitative_cfg.get("min_foreground_ratio", 0.005)
        if min_foreground_ratio is None
        else min_foreground_ratio
    )
    max_foreground_ratio = float(
        qualitative_cfg.get("max_foreground_ratio", 0.15)
        if max_foreground_ratio is None
        else max_foreground_ratio
    )
    selection_seed = (
        qualitative_cfg.get("selection_seed", None) if selection_seed is None else selection_seed
    )
    selection_seed = None if selection_seed is None else int(selection_seed)
    selection_rng = None if selection_seed is None else np.random.default_rng(selection_seed)
    threshold = float(config.get("inference", {}).get("threshold", 0.5) if threshold is None else threshold)
    device = resolve_device(device_name or str(config["train"].get("device", "auto")))
    max_checkpoints = qualitative_cfg.get("max_checkpoints") if max_checkpoints is None else max_checkpoints
    max_checkpoints = None if max_checkpoints is None else int(max_checkpoints)

    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        message = f"Qualitative data root is missing images/ or masks/: {data_root}"
        if logger:
            logger.warning(message)
        return {"skipped": True, "reason": message}

    pairs, diagnostics = discover_image_mask_pairs(
        images_dir,
        masks_dir,
        config["data"]["image_extensions"],
    )
    if diagnostics["missing_masks"] and logger:
        logger.warning("Qualitative evaluation missing masks for %s images.", len(diagnostics["missing_masks"]))
    if diagnostics["missing_images"] and logger:
        logger.warning("Qualitative evaluation found %s masks without images.", len(diagnostics["missing_images"]))
    if not pairs:
        message = f"No qualitative image/mask pairs found under {data_root}"
        if logger:
            logger.warning(message)
        return {"skipped": True, "reason": message}

    checkpoints = discover_manifest_checkpoints(run_dir, max_checkpoints=max_checkpoints)
    if not checkpoints:
        raise RuntimeError(f"No manifest checkpoints found under {run_dir}.")

    if logger:
        logger.info(
            "Running qualitative evaluation on %s image(s) with %s checkpoint(s).",
            len(pairs),
            len(checkpoints),
        )

    patch_size = int(config["data"]["patch_size"])
    stride = int(config["data"]["stride"])
    mask_threshold = int(config["data"]["mask_threshold"])
    use_amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    metric_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    selected_crops: dict[str, SelectedCrop] = {}
    image_payloads: list[dict[str, Any]] = []

    for image_path, mask_path in tqdm(pairs, desc="Qualitative images"):
        with Image.open(image_path) as image:
            image_array = np.array(image.convert("RGB"))
        with Image.open(mask_path) as mask:
            mask_array = np.array(mask.convert("L"), dtype=np.uint8)
        if mask_array.shape[:2] != image_array.shape[:2]:
            raise ValueError(
                f"Image and mask shapes differ for {image_path.name}: "
                f"{image_array.shape[:2]} vs {mask_array.shape[:2]}"
            )

        crop = select_qualitative_crop(
            mask_array=mask_array,
            patch_size=patch_size,
            stride=stride,
            crop_patch_grid=(int(crop_patch_grid[0]), int(crop_patch_grid[1])),
            mask_threshold=mask_threshold,
            min_foreground_ratio=min_foreground_ratio,
            max_foreground_ratio=max_foreground_ratio,
            rng=selection_rng,
        )
        selected_crops[image_path.stem] = crop
        crop_rows.append(crop_row(image_path, mask_path, crop, selection_seed))
        image_crop = image_array[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        target_crop = (mask_array[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] > mask_threshold).astype(
            np.uint8
        )
        panels = [
            make_panel(f"{image_path.stem}\nsource", image_crop),
            make_panel("ground truth", create_overlay(image_crop, target_crop * 255), target_crop),
        ]
        image_payloads.append(
            {
                "image_path": image_path,
                "image_crop": image_crop,
                "target_crop": target_crop,
                "crop": crop,
                "panels": panels,
            }
        )

    for entry in tqdm(checkpoints, desc="Qualitative checkpoints"):
        model = build_model(config["model"]).to(device)
        load_checkpoint(entry.path, model, map_location=device)
        model.eval()
        for payload in image_payloads:
            with Image.open(payload["image_path"]) as image:
                image_array = np.array(image.convert("RGB"))
            probabilities = predict_crop_probabilities(
                model=model,
                image_array=image_array,
                crop=payload["crop"],
                config=config,
                device=device,
                use_amp=use_amp,
            )
            prediction_mask = (probabilities >= threshold).astype(np.uint8)
            metric_rows.append(
                metric_row(
                    payload["image_path"],
                    entry,
                    payload["crop"],
                    prediction_mask,
                    payload["target_crop"],
                )
            )
            payload["panels"].append(
                make_panel(
                    _checkpoint_label(entry),
                    create_overlay(payload["image_crop"], prediction_mask * 255),
                    prediction_mask,
                )
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for payload in image_payloads:
        save_grid(grids_dir / f"{payload['image_path'].stem}.png", payload["panels"])

    save_csv(output_dir / "eval_metrics.csv", metric_rows)
    save_csv(output_dir / "selected_crops.csv", crop_rows)
    return {
        "skipped": False,
        "output_dir": str(output_dir),
        "num_images": len(pairs),
        "num_checkpoints": len(checkpoints),
        "num_metric_rows": len(metric_rows),
        "selected_crops": selected_crops,
    }


def main() -> None:
    args = parse_args()
    run_qualitative_evaluation(
        run_dir=args.run_dir,
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        crop_patch_grid=tuple(args.crop_patch_grid) if args.crop_patch_grid else None,
        min_foreground_ratio=args.min_foreground_ratio,
        max_foreground_ratio=args.max_foreground_ratio,
        selection_seed=args.selection_seed,
        threshold=args.threshold,
        device_name=args.device,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == "__main__":
    main()
