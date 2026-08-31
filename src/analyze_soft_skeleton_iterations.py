from __future__ import annotations

import argparse
import math
import statistics
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from src.data.soft_cldice_iterations import required_soft_skeleton_iterations
from src.metrics.segmentation import _soft_erode, _soft_open
from src.utils.io import ensure_dir, save_csv, save_json, save_mask_image


Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep production soft-clDice skeleton iterations over full-resolution "
            "binary masks and compare them with the maximum iteration and Zhang."
        )
    )
    parser.add_argument("--mask-dir", default="data/loci_masks")
    parser.add_argument(
        "--output-dir", default="outputs/soft-skeleton-iteration-analysis"
    )
    parser.add_argument(
        "--mask-name",
        action="append",
        default=None,
        help="Process exactly this filename; repeat for multiple selected masks.",
    )
    parser.add_argument(
        "--save-full-resolution",
        action="store_true",
        help="Save ground truth and every tested soft skeleton at full resolution.",
    )
    parser.add_argument("--min-iterations", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--iteration-step", type=int, default=10)
    parser.add_argument(
        "--visual-iterations",
        default="30,50,70,90,110,130,150",
        help="Comma-separated iterations shown in each per-image contact sheet.",
    )
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, or a device such as cuda:0."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Interior tile size; each tile gets an exact max-iteration halo.",
    )
    parser.add_argument(
        "--zhang-tolerance",
        type=int,
        default=2,
        help="Pixel radius for tolerant soft-vs-Zhang precision and recall.",
    )
    parser.add_argument(
        "--capture-threshold",
        type=float,
        default=0.999,
        help="Worst-image fraction of the max-iteration skeleton to recommend.",
    )
    parser.add_argument("--preview-size", type=int, default=640)
    return parser.parse_args()


def _parse_visual_iterations(value: str, minimum: int, maximum: int) -> list[int]:
    iterations = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [item for item in iterations if item < minimum or item > maximum]
    if invalid:
        raise ValueError(
            f"Visual iterations must be within [{minimum}, {maximum}], got {invalid}."
        )
    return iterations


def _iteration_values(minimum: int, maximum: int, step: int) -> list[int]:
    if minimum < 0 or maximum < minimum:
        raise ValueError("Iteration bounds must satisfy 0 <= minimum <= maximum.")
    if step <= 0:
        raise ValueError("iteration_step must be positive.")
    values = list(range(minimum, maximum + 1, step))
    if values[-1] != maximum:
        values.append(maximum)
    return values


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({value}) but is unavailable.")
    return device


def discover_masks(
    mask_dir: str | Path,
    selected_names: Iterable[str] | None = None,
) -> list[Path]:
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    paths: list[Path] = []
    for path in sorted(mask_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError):
            continue
        paths.append(path)
    if not paths:
        raise ValueError(f"No readable mask images found in {mask_dir}.")
    if selected_names is None:
        return paths
    available = {path.name: path for path in paths}
    requested = list(dict.fromkeys(selected_names))
    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Selected mask filenames were not found in {mask_dir}: {missing}"
        )
    return [available[name] for name in requested]


def load_binary_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > threshold


def _tile_starts(length: int, tile_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, tile_size):
        yield start, min(length, start + tile_size)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sweep_soft_skeleton(
    mask: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    iteration_step: int = 1,
    visual_iterations: Iterable[int],
    tile_size: int,
    device: torch.device,
) -> tuple[
    dict[int, dict[str, int]],
    dict[int, np.ndarray],
    float,
    dict[int, float],
]:
    """Run exact haloed tiles and benchmark cumulative skeletonization kernels."""
    evaluated = _iteration_values(minimum, maximum, iteration_step)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    height, width = mask.shape
    halo = maximum + 1
    requested = set(visual_iterations)
    snapshots = {
        iteration: np.zeros_like(mask, dtype=bool) for iteration in requested
    }
    totals = {
        iteration: {"skeleton_pixels": 0, "eroded_foreground_pixels": 0}
        for iteration in evaluated
    }
    step_seconds = np.zeros(maximum + 1, dtype=np.float64)
    cuda_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []

    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for y0, y1 in _tile_starts(height, tile_size):
            for x0, x1 in _tile_starts(width, tile_size):
                sy0, sy1 = max(0, y0 - halo), min(height, y1 + halo)
                sx0, sx1 = max(0, x0 - halo), min(width, x1 + halo)
                tile = torch.from_numpy(mask[sy0:sy1, sx0:sx1]).to(
                    device=device, dtype=torch.float32
                )[None, None]
                if device.type == "cuda":
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_start.record()
                    soft_skeleton = torch.relu(tile - _soft_open(tile))
                    event_end.record()
                    cuda_events.append((0, event_start, event_end))
                else:
                    operation_started = time.perf_counter()
                    soft_skeleton = torch.relu(tile - _soft_open(tile))
                    step_seconds[0] += time.perf_counter() - operation_started
                crop = (slice(y0 - sy0, y1 - sy0), slice(x0 - sx0, x1 - sx0))
                for iteration in range(maximum + 1):
                    if iteration in totals:
                        skeleton_crop = soft_skeleton[0, 0][crop]
                        mask_crop = tile[0, 0][crop]
                        totals[iteration]["skeleton_pixels"] += int(
                            torch.count_nonzero(skeleton_crop).item()
                        )
                        totals[iteration]["eroded_foreground_pixels"] += int(
                            torch.count_nonzero(mask_crop).item()
                        )
                        if iteration in requested:
                            snapshots[iteration][y0:y1, x0:x1] = (
                                skeleton_crop.detach().cpu().numpy() > 0.5
                            )
                    if iteration == maximum:
                        break
                    if device.type == "cuda":
                        event_start = torch.cuda.Event(enable_timing=True)
                        event_end = torch.cuda.Event(enable_timing=True)
                        event_start.record()
                        tile = _soft_erode(tile)
                        delta = torch.relu(tile - _soft_open(tile))
                        soft_skeleton = soft_skeleton + torch.relu(
                            delta - soft_skeleton * delta
                        )
                        event_end.record()
                        cuda_events.append((iteration + 1, event_start, event_end))
                    else:
                        operation_started = time.perf_counter()
                        tile = _soft_erode(tile)
                        delta = torch.relu(tile - _soft_open(tile))
                        soft_skeleton = soft_skeleton + torch.relu(
                            delta - soft_skeleton * delta
                        )
                        step_seconds[iteration + 1] += (
                            time.perf_counter() - operation_started
                        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        for iteration, event_start, event_end in cuda_events:
            step_seconds[iteration] += event_start.elapsed_time(event_end) / 1000.0
    cumulative = np.cumsum(step_seconds)
    kernel_seconds = {
        iteration: float(cumulative[iteration]) for iteration in evaluated
    }
    return totals, snapshots, elapsed, kernel_seconds


def skeleton_similarity(
    soft: np.ndarray, hard: np.ndarray, tolerance: int
) -> dict[str, float | int]:
    soft = np.asarray(soft, dtype=bool)
    hard = np.asarray(hard, dtype=bool)
    soft_pixels = int(soft.sum())
    hard_pixels = int(hard.sum())
    intersection = int(np.count_nonzero(soft & hard))
    denominator = soft_pixels + hard_pixels
    raw_dice = 1.0 if denominator == 0 else 2 * intersection / denominator
    if tolerance > 0:
        structure = np.ones((3, 3), dtype=bool)
        hard_region = binary_dilation(hard, structure=structure, iterations=tolerance)
        soft_region = binary_dilation(soft, structure=structure, iterations=tolerance)
    else:
        hard_region, soft_region = hard, soft
    matched_soft = int(np.count_nonzero(soft & hard_region))
    matched_hard = int(np.count_nonzero(hard & soft_region))
    both_empty = soft_pixels == 0 and hard_pixels == 0
    precision = 1.0 if both_empty else matched_soft / soft_pixels if soft_pixels else 0.0
    recall = 1.0 if both_empty else matched_hard / hard_pixels if hard_pixels else 0.0
    tolerant_denominator = precision + recall
    return {
        "soft_pixels": soft_pixels,
        "zhang_pixels": hard_pixels,
        "raw_dice": raw_dice,
        "tolerant_precision": precision,
        "tolerant_recall": recall,
        "tolerant_f1": (
            0.0
            if tolerant_denominator == 0
            else 2 * precision * recall / tolerant_denominator
        ),
    }


def _preview_shape(shape: tuple[int, int], maximum_size: int) -> tuple[int, int]:
    height, width = shape
    scale = min(1.0, maximum_size / max(height, width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _binary_max_preview(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downsample while retaining every one-pixel skeleton branch."""
    output_width, output_height = size
    ys, xs = np.nonzero(mask)
    output = np.zeros((output_height, output_width), dtype=np.uint8)
    if len(ys):
        out_y = np.minimum((ys * output_height) // mask.shape[0], output_height - 1)
        out_x = np.minimum((xs * output_width) // mask.shape[1], output_width - 1)
        output[out_y, out_x] = 255
    return output


def _panel(image: np.ndarray, label: str) -> Image.Image:
    panel = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return panel


def save_contact_sheet(
    path: Path,
    mask: np.ndarray,
    zhang: np.ndarray,
    snapshots: dict[int, np.ndarray],
    preview_size: int,
) -> None:
    size = _preview_shape(mask.shape, preview_size)
    mask_preview = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BOX)
    )
    panels = [
        _panel(mask_preview, "binary loci mask"),
        _panel(_binary_max_preview(zhang, size), "Zhang skeleton"),
    ]
    panels.extend(
        _panel(_binary_max_preview(item, size), f"soft skeleton: {iteration}")
        for iteration, item in sorted(snapshots.items())
    )
    columns = 3
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * size[0], rows * size[1]))
    for index, panel in enumerate(panels):
        sheet.paste(
            panel, ((index % columns) * size[0], (index // columns) * size[1])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_full_resolution_outputs(
    root: Path,
    source_path: Path,
    mask: np.ndarray,
    snapshots: dict[int, np.ndarray],
) -> dict[str, Any]:
    image_dir = ensure_dir(root / source_path.stem)
    ground_truth_name = "ground_truth.png"
    save_mask_image(image_dir / ground_truth_name, mask.astype(np.uint8) * 255)
    skeleton_files: dict[str, str] = {}
    for iteration, skeleton in sorted(snapshots.items()):
        filename = f"soft_skeleton_{iteration}.png"
        save_mask_image(image_dir / filename, skeleton.astype(np.uint8) * 255)
        skeleton_files[str(iteration)] = str(image_dir / filename)
    return {
        "directory": str(image_dir),
        "ground_truth": str(image_dir / ground_truth_name),
        "soft_skeletons": skeleton_files,
    }


def analyze_mask(
    path: Path,
    *,
    minimum: int,
    maximum: int,
    iteration_step: int,
    visual_iterations: list[int],
    threshold: int,
    tile_size: int,
    device: torch.device,
    zhang_tolerance: int,
    preview_size: int,
    visual_dir: Path,
    full_resolution_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = load_binary_mask(path, threshold)
    foreground_pixels = int(mask.sum())
    required_iterations = required_soft_skeleton_iterations(mask)
    evaluated = _iteration_values(minimum, maximum, iteration_step)
    snapshot_iterations = set(visual_iterations) | {maximum}
    if full_resolution_dir is not None:
        snapshot_iterations.update(evaluated)
    totals, snapshots, elapsed, kernel_seconds = sweep_soft_skeleton(
        mask,
        minimum=minimum,
        maximum=maximum,
        iteration_step=iteration_step,
        visual_iterations=snapshot_iterations,
        tile_size=tile_size,
        device=device,
    )
    reference_pixels = totals[maximum]["skeleton_pixels"]
    zhang = skeletonize(mask, method="zhang")
    similarity = skeleton_similarity(snapshots[maximum], zhang, zhang_tolerance)
    visual_path = visual_dir / f"{path.stem}.png"
    save_contact_sheet(
        visual_path,
        mask,
        zhang,
        {item: snapshots[item] for item in visual_iterations},
        preview_size,
    )
    full_resolution_artifacts = None
    if full_resolution_dir is not None:
        full_resolution_artifacts = save_full_resolution_outputs(
            full_resolution_dir,
            path,
            mask,
            {item: snapshots[item] for item in evaluated},
        )

    rows = []
    for evaluated_index, iteration in enumerate(evaluated):
        pixels = totals[iteration]["skeleton_pixels"]
        previous_iteration = evaluated[evaluated_index - 1] if evaluated_index else None
        previous = (
            totals[previous_iteration]["skeleton_pixels"]
            if previous_iteration is not None
            else 0
        )
        captured = 1.0 if reference_pixels == 0 else pixels / reference_pixels
        rows.append(
            {
                "filename": path.name,
                "width": mask.shape[1],
                "height": mask.shape[0],
                "foreground_pixels": foreground_pixels,
                "required_iterations": required_iterations,
                "iterations": iteration,
                "skeleton_pixels": pixels,
                "previous_tested_iterations": previous_iteration,
                "new_skeleton_pixels_since_previous_tested": pixels - previous,
                "missing_vs_max_iteration": reference_pixels - pixels,
                "capture_vs_max_iteration": captured,
                "dice_vs_max_iteration": (
                    1.0
                    if pixels + reference_pixels == 0
                    else 2 * pixels / (pixels + reference_pixels)
                ),
                "eroded_foreground_pixels": totals[iteration][
                    "eroded_foreground_pixels"
                ],
                "relative_iteration_compute": (iteration + 1) / (maximum + 1),
                "skeletonization_kernel_seconds": kernel_seconds[iteration],
                "sweep_wall_seconds_at_max_iteration": elapsed,
            }
        )
    image_summary = {
        "filename": path.name,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "foreground_pixels": foreground_pixels,
        "required_iterations": required_iterations,
        "max_tested_iterations": maximum,
        "max_tested_is_complete": required_iterations <= maximum,
        "sweep_seconds": elapsed,
        "kernel_benchmark": {
            "device": str(device),
            "excludes": "mask transfer, metric reduction, PNG/CSV I/O, and autograd",
            "cumulative_seconds": kernel_seconds,
        },
        "soft_max_vs_zhang": similarity,
        "visual": str(visual_path),
        "full_resolution_artifacts": full_resolution_artifacts,
    }
    return rows, image_summary


def aggregate_iteration_rows(
    rows: list[dict[str, Any]],
    minimum: int,
    maximum: int,
    iteration_step: int,
) -> list[dict[str, Any]]:
    aggregate = []
    for iteration in _iteration_values(minimum, maximum, iteration_step):
        selected = [row for row in rows if row["iterations"] == iteration]
        captures = [float(row["capture_vs_max_iteration"]) for row in selected]
        skeleton_pixels = sum(int(row["skeleton_pixels"]) for row in selected)
        reference_pixels = skeleton_pixels + sum(
            int(row["missing_vs_max_iteration"]) for row in selected
        )
        kernel_seconds = sum(
            float(row["skeletonization_kernel_seconds"]) for row in selected
        )
        aggregate.append(
            {
                "iterations": iteration,
                "images": len(selected),
                "minimum_capture_vs_max_iteration": min(captures),
                "10th_percentile_capture_vs_max_iteration": float(
                    np.quantile(captures, 0.1)
                ),
                "mean_capture_vs_max_iteration": statistics.fmean(captures),
                "global_pixel_capture_vs_max_iteration": (
                    1.0
                    if reference_pixels == 0
                    else skeleton_pixels / reference_pixels
                ),
                "images_fully_matching_max_iteration": sum(
                    int(row["missing_vs_max_iteration"] == 0) for row in selected
                ),
                "total_missing_vs_max_iteration": sum(
                    int(row["missing_vs_max_iteration"]) for row in selected
                ),
                "total_new_skeleton_pixels_since_previous_tested": sum(
                    int(row["new_skeleton_pixels_since_previous_tested"])
                    for row in selected
                ),
                "relative_iteration_compute": (iteration + 1) / (maximum + 1),
                "full_dataset_skeletonization_kernel_seconds": kernel_seconds,
            }
        )
    maximum_seconds = float(
        aggregate[-1]["full_dataset_skeletonization_kernel_seconds"]
    )
    for row in aggregate:
        seconds = float(row["full_dataset_skeletonization_kernel_seconds"])
        row["measured_time_relative_to_max"] = (
            1.0 if maximum_seconds == 0 else seconds / maximum_seconds
        )
        row["measured_seconds_saved_vs_max"] = maximum_seconds - seconds
    return aggregate


def run_analysis(
    mask_dir: str | Path,
    output_dir: str | Path,
    *,
    minimum: int = 30,
    maximum: int = 150,
    iteration_step: int = 10,
    visual_iterations: list[int] | None = None,
    mask_names: Iterable[str] | None = None,
    save_full_resolution: bool = False,
    threshold: int = 127,
    device: torch.device | None = None,
    tile_size: int = 2048,
    zhang_tolerance: int = 2,
    capture_threshold: float = 0.999,
    preview_size: int = 640,
) -> dict[str, Any]:
    if not 0.0 < capture_threshold <= 1.0:
        raise ValueError("capture_threshold must be in (0, 1].")
    if zhang_tolerance < 0:
        raise ValueError("zhang_tolerance must be non-negative.")
    device = device or _resolve_device("auto")
    evaluated = _iteration_values(minimum, maximum, iteration_step)
    visual_iterations = visual_iterations or [minimum, maximum]
    paths = discover_masks(mask_dir, mask_names)
    output_dir = ensure_dir(output_dir)
    visual_dir = ensure_dir(output_dir / "visuals")
    full_resolution_dir = (
        ensure_dir(output_dir / "full_resolution") if save_full_resolution else None
    )
    all_rows: list[dict[str, Any]] = []
    image_summaries = []
    print(f"Analyzing {len(paths)} masks on {device}.", flush=True)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        rows, image_summary = analyze_mask(
            path,
            minimum=minimum,
            maximum=maximum,
            iteration_step=iteration_step,
            visual_iterations=visual_iterations,
            threshold=threshold,
            tile_size=tile_size,
            device=device,
            zhang_tolerance=zhang_tolerance,
            preview_size=preview_size,
            visual_dir=visual_dir,
            full_resolution_dir=full_resolution_dir,
        )
        all_rows.extend(rows)
        image_summaries.append(image_summary)
        save_csv(output_dir / "per_image_iterations.csv", all_rows)

    aggregate = aggregate_iteration_rows(
        all_rows, minimum, maximum, iteration_step
    )
    recommended = next(
        (
            int(row["iterations"])
            for row in aggregate
            if row["minimum_capture_vs_max_iteration"] >= capture_threshold
        ),
        maximum,
    )
    required_maximum = max(int(item["required_iterations"]) for item in image_summaries)
    complete_images = sum(
        int(item["max_tested_is_complete"]) for item in image_summaries
    )
    summary = {
        "mask_dir": str(Path(mask_dir)),
        "output_dir": str(output_dir),
        "device": str(device),
        "mask_count": len(paths),
        "selected_mask_names": [path.name for path in paths],
        "save_full_resolution": save_full_resolution,
        "min_iterations": minimum,
        "max_iterations": maximum,
        "iteration_step": iteration_step,
        "tested_iterations": evaluated,
        "capture_threshold": capture_threshold,
        "recommended_iterations_relative_to_max_tested": recommended,
        "maximum_exact_required_iterations": required_maximum,
        "images_complete_at_max_tested": complete_images,
        "all_images_complete_at_max_tested": complete_images == len(paths),
        "interpretation": (
            "The recommendation is the cheapest iteration whose worst-image soft "
            "skeleton captures the requested fraction of the max-tested skeleton. "
            "It is conclusive only when all_images_complete_at_max_tested is true; "
            "maximum_exact_required_iterations is the crisp-target geometric bound."
        ),
        "performance_benchmark": {
            "measurement": "cumulative soft-skeletonization kernel time",
            "device": str(device),
            "scope": "all masks, original resolution, exact haloed tiles",
            "excludes": (
                "mask transfer, metric reduction, Zhang skeletonization, "
                "PNG/CSV I/O, and autograd"
            ),
            "note": "Training forward/backward wall time will be higher.",
        },
        "zhang_comparison": {
            "tolerance_pixels": zhang_tolerance,
            "minimum_tolerant_f1": min(
                float(item["soft_max_vs_zhang"]["tolerant_f1"])
                for item in image_summaries
            ),
            "mean_tolerant_f1": statistics.fmean(
                float(item["soft_max_vs_zhang"]["tolerant_f1"])
                for item in image_summaries
            ),
            "note": (
                "Zhang and differentiable morphological skeletons need not occupy "
                "identical centerline pixels; tolerant F1 is a visual sanity check, "
                "not the iteration-selection criterion."
            ),
        },
        "images": image_summaries,
        "artifacts": {
            "per_image_iterations": "per_image_iterations.csv",
            "aggregate_iterations": "aggregate_iterations.csv",
            "summary": "summary.json",
            "visuals": "visuals/",
            "full_resolution": (
                "full_resolution/" if save_full_resolution else None
            ),
        },
    }
    save_csv(output_dir / "per_image_iterations.csv", all_rows)
    save_csv(output_dir / "aggregate_iterations.csv", aggregate)
    save_json(output_dir / "summary.json", summary)
    print(
        f"Recommended relative to iteration {maximum}: {recommended}; exact maximum "
        f"required by these binary masks: {required_maximum}.",
        flush=True,
    )
    print(f"Artifacts written to {output_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.min_iterations < 0 or args.max_iterations < args.min_iterations:
        raise ValueError("Require 0 <= --min-iterations <= --max-iterations.")
    visual_iterations = _parse_visual_iterations(
        args.visual_iterations, args.min_iterations, args.max_iterations
    )
    run_analysis(
        args.mask_dir,
        args.output_dir,
        minimum=args.min_iterations,
        maximum=args.max_iterations,
        iteration_step=args.iteration_step,
        visual_iterations=visual_iterations,
        mask_names=args.mask_name,
        save_full_resolution=args.save_full_resolution,
        threshold=args.threshold,
        device=_resolve_device(args.device),
        tile_size=args.tile_size,
        zhang_tolerance=args.zhang_tolerance,
        capture_threshold=args.capture_threshold,
        preview_size=args.preview_size,
    )


if __name__ == "__main__":
    main()
