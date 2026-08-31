from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.benchmark_cldice import (
    benchmark_device,
    create_overlap_image,
    load_binary_mask,
    skeleton_similarity,
)
from src.metrics.segmentation import cldice_score_from_skeletons
from src.patching import _compute_positions, crop_and_pad_array
from src.utils.io import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark batched hard clDice on deterministic foreground patches "
            "using CPU and CUDA."
        )
    )
    parser.add_argument("--prediction-mask", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/cldice-patch-benchmark",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Grid stride used before seeded foreground-patch selection.",
    )
    parser.add_argument("--num-patches", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of patches skeletonized together, matching validation batching.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--foreground-value", type=int, default=None)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--smooth", type=float, default=1e-6)
    return parser.parse_args()


def select_foreground_patches(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    num_patches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction and target mask dimensions differ: "
            f"prediction={prediction_mask.shape}, target={target_mask.shape}."
        )
    if prediction_mask.ndim != 2:
        raise ValueError("Patch selection expects two-dimensional masks.")
    for name, value in (
        ("patch_size", patch_size),
        ("stride", stride),
        ("num_patches", num_patches),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    height, width = prediction_mask.shape
    foreground = np.asarray(prediction_mask | target_mask, dtype=bool)
    candidates: list[dict[str, int]] = []
    for y in _compute_positions(height, patch_size, stride):
        for x in _compute_positions(width, patch_size, stride):
            patch = crop_and_pad_array(foreground, x, y, patch_size)
            foreground_pixels = int(np.count_nonzero(patch))
            if foreground_pixels:
                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "foreground_pixels": foreground_pixels,
                    }
                )
    if not candidates:
        raise ValueError("No foreground-containing patches were found.")

    selected_count = min(num_patches, len(candidates))
    random = np.random.default_rng(seed)
    selected_indices = sorted(
        int(index)
        for index in random.choice(
            len(candidates),
            size=selected_count,
            replace=False,
        )
    )
    selected = [candidates[index] for index in selected_indices]
    prediction_patches = np.stack(
        [
            crop_and_pad_array(
                prediction_mask,
                record["x"],
                record["y"],
                patch_size,
            )
            for record in selected
        ]
    ).astype(bool)
    target_patches = np.stack(
        [
            crop_and_pad_array(
                target_mask,
                record["x"],
                record["y"],
                patch_size,
            )
            for record in selected
        ]
    ).astype(bool)
    return prediction_patches, target_patches, selected


def _as_patch_stack(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=bool)
    return array[None, ...] if array.ndim == 2 else array


def benchmark_patch_device(
    prediction_patches: np.ndarray,
    target_patches: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    repeats: int,
    smooth: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    num_patches = int(prediction_patches.shape[0])
    repeat_timings: list[float] = []
    repeat_transfers: list[float] = []
    cldice_weighted_total = 0.0
    prediction_skeletons = target_skeletons = None
    maximum_peak_bytes = 0
    maximum_extra_peak_bytes = 0
    cuda_name = None

    for repeat in range(1, repeats + 1):
        print(f"  {device} patch pass {repeat}/{repeats}...", flush=True)
        elapsed = 0.0
        transfer = 0.0
        weighted_score = 0.0
        prediction_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        for start in range(0, num_patches, batch_size):
            stop = min(start + batch_size, num_patches)
            result, prediction_part, target_part = benchmark_device(
                prediction_patches[start:stop],
                target_patches[start:stop],
                device=device,
                repeats=1,
                smooth=smooth,
            )
            count = stop - start
            elapsed += float(result["median_seconds"])
            transfer += float(result["transfer_seconds"])
            weighted_score += float(result["cldice"]) * count
            prediction_parts.append(_as_patch_stack(prediction_part))
            target_parts.append(_as_patch_stack(target_part))
            maximum_peak_bytes = max(
                maximum_peak_bytes,
                int(result.get("peak_cuda_allocated_bytes", 0)),
            )
            maximum_extra_peak_bytes = max(
                maximum_extra_peak_bytes,
                int(result.get("peak_extra_cuda_allocated_bytes", 0)),
            )
            cuda_name = result.get("cuda_name", cuda_name)
        repeat_timings.append(elapsed)
        repeat_transfers.append(transfer)
        cldice_weighted_total = weighted_score
        prediction_skeletons = np.concatenate(prediction_parts, axis=0)
        target_skeletons = np.concatenate(target_parts, axis=0)

    if prediction_skeletons is None or target_skeletons is None:
        raise RuntimeError("The patch benchmark did not produce skeletons.")

    median_seconds = statistics.median(repeat_timings)
    result: dict[str, Any] = {
        "device": str(device),
        "num_patches": num_patches,
        "batch_size": batch_size,
        "cldice": cldice_weighted_total / num_patches,
        "transfer_seconds_per_repeat": repeat_transfers,
        "timings_seconds": repeat_timings,
        "mean_seconds": statistics.mean(repeat_timings),
        "median_seconds": median_seconds,
        "min_seconds": min(repeat_timings),
        "max_seconds": max(repeat_timings),
        "median_seconds_per_patch": median_seconds / num_patches,
        "median_patches_per_second": num_patches / median_seconds,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_name": cuda_name,
                "maximum_peak_cuda_allocated_bytes": maximum_peak_bytes,
                "maximum_peak_extra_cuda_allocated_bytes": maximum_extra_peak_bytes,
            }
        )
    return result, prediction_skeletons, target_skeletons


def _per_patch_cldice(
    prediction_skeletons: np.ndarray,
    target_skeletons: np.ndarray,
    prediction_patches: np.ndarray,
    target_patches: np.ndarray,
    smooth: float,
) -> list[float]:
    return [
        cldice_score_from_skeletons(
            torch.from_numpy(prediction_skeletons[index]),
            torch.from_numpy(target_skeletons[index]),
            torch.from_numpy(prediction_patches[index]),
            torch.from_numpy(target_patches[index]),
            smooth=smooth,
        )
        for index in range(len(prediction_patches))
    ]


def _contact_sheet(images: list[np.ndarray], columns: int = 5) -> np.ndarray:
    if not images:
        raise ValueError("Cannot create a contact sheet without images.")
    first = np.asarray(images[0])
    height, width = first.shape[:2]
    rows = (len(images) + columns - 1) // columns
    output_shape = (
        (rows * height, columns * width, first.shape[2])
        if first.ndim == 3
        else (rows * height, columns * width)
    )
    output = np.zeros(output_shape, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        output[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
            ...,
        ] = np.asarray(image, dtype=np.uint8)
    return output


def _save_contact_sheet(path: Path, images: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_contact_sheet(images)).save(path)


def run_patch_benchmark(
    prediction_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: int = 512,
    stride: int = 256,
    num_patches: int = 50,
    batch_size: int = 8,
    seed: int = 42,
    threshold: int = 127,
    foreground_value: int | None = None,
    cuda_device: str = "cuda:0",
    repeats: int = 1,
    smooth: float = 1e-6,
) -> dict[str, Any]:
    prediction_path = Path(prediction_path)
    target_path = Path(target_path)
    output_dir = ensure_dir(output_dir)
    prediction_mask = load_binary_mask(
        prediction_path,
        threshold=threshold,
        foreground_value=foreground_value,
    )
    target_mask = load_binary_mask(
        target_path,
        threshold=threshold,
        foreground_value=foreground_value,
    )
    prediction_patches, target_patches, patch_records = select_foreground_patches(
        prediction_mask,
        target_mask,
        patch_size=patch_size,
        stride=stride,
        num_patches=num_patches,
        seed=seed,
    )
    actual_num_patches = len(patch_records)
    print(
        f"Selected {actual_num_patches}/{num_patches} foreground patches "
        f"of size {patch_size}x{patch_size} with seed {seed}.",
        flush=True,
    )

    cpu_result, prediction_cpu, target_cpu = benchmark_patch_device(
        prediction_patches,
        target_patches,
        device=torch.device("cpu"),
        batch_size=batch_size,
        repeats=repeats,
        smooth=smooth,
    )
    cuda = torch.device(cuda_device)
    if cuda.type != "cuda":
        raise ValueError("--cuda-device must name a CUDA device, such as cuda:0.")
    try:
        gpu_result, prediction_gpu, target_gpu = benchmark_patch_device(
            prediction_patches,
            target_patches,
            device=cuda,
            batch_size=batch_size,
            repeats=repeats,
            smooth=smooth,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA ran out of memory during the patch benchmark. "
            "Reduce --batch-size."
        ) from error

    cpu_scores = _per_patch_cldice(
        prediction_cpu,
        target_cpu,
        prediction_patches,
        target_patches,
        smooth,
    )
    gpu_scores = _per_patch_cldice(
        prediction_gpu,
        target_gpu,
        prediction_patches,
        target_patches,
        smooth,
    )
    patch_rows = []
    prediction_similarities = []
    target_similarities = []
    for index, record in enumerate(patch_records):
        prediction_similarity = skeleton_similarity(
            prediction_cpu[index], prediction_gpu[index]
        )
        target_similarity = skeleton_similarity(target_cpu[index], target_gpu[index])
        prediction_similarities.append(prediction_similarity)
        target_similarities.append(target_similarity)
        patch_rows.append(
            {
                "patch_index": index,
                **record,
                "prediction_foreground_pixels": int(
                    np.count_nonzero(prediction_patches[index])
                ),
                "target_foreground_pixels": int(
                    np.count_nonzero(target_patches[index])
                ),
                "cpu_cldice": cpu_scores[index],
                "gpu_cldice": gpu_scores[index],
                "cldice_absolute_difference": abs(
                    cpu_scores[index] - gpu_scores[index]
                ),
                "prediction_skeleton_dice": prediction_similarity["dice"],
                "prediction_skeleton_different_pixels": prediction_similarity[
                    "different_pixels"
                ],
                "target_skeleton_dice": target_similarity["dice"],
                "target_skeleton_different_pixels": target_similarity[
                    "different_pixels"
                ],
            }
        )

    artifacts = {
        "patch_metrics": "patch_metrics.csv",
        "prediction_input_patches": "prediction_input_patches.png",
        "target_input_patches": "target_input_patches.png",
        "cpu_prediction_skeletons": "cpu_prediction_skeletons.png",
        "gpu_prediction_skeletons": "gpu_prediction_skeletons.png",
        "prediction_cpu_gpu_overlap": "prediction_cpu_vs_gpu_skeletons.png",
        "target_cpu_gpu_overlap": "target_cpu_vs_gpu_skeletons.png",
        "gpu_prediction_target_overlap": "gpu_prediction_vs_target_skeletons.png",
    }
    save_csv(output_dir / artifacts["patch_metrics"], patch_rows)
    _save_contact_sheet(
        output_dir / artifacts["prediction_input_patches"],
        [patch * 255 for patch in prediction_patches],
    )
    _save_contact_sheet(
        output_dir / artifacts["target_input_patches"],
        [patch * 255 for patch in target_patches],
    )
    _save_contact_sheet(
        output_dir / artifacts["cpu_prediction_skeletons"],
        [patch * 255 for patch in prediction_cpu],
    )
    _save_contact_sheet(
        output_dir / artifacts["gpu_prediction_skeletons"],
        [patch * 255 for patch in prediction_gpu],
    )
    _save_contact_sheet(
        output_dir / artifacts["prediction_cpu_gpu_overlap"],
        [
            create_overlap_image(prediction_cpu[index], prediction_gpu[index])
            for index in range(actual_num_patches)
        ],
    )
    _save_contact_sheet(
        output_dir / artifacts["target_cpu_gpu_overlap"],
        [
            create_overlap_image(target_cpu[index], target_gpu[index])
            for index in range(actual_num_patches)
        ],
    )
    _save_contact_sheet(
        output_dir / artifacts["gpu_prediction_target_overlap"],
        [
            create_overlap_image(prediction_gpu[index], target_gpu[index])
            for index in range(actual_num_patches)
        ],
    )

    speedup = cpu_result["median_seconds"] / gpu_result["median_seconds"]
    cldice_differences = [
        abs(cpu_score - gpu_score)
        for cpu_score, gpu_score in zip(cpu_scores, gpu_scores)
    ]
    summary = {
        "prediction_mask": str(prediction_path),
        "target_mask": str(target_path),
        "output_dir": str(output_dir),
        "patch_size": patch_size,
        "stride": stride,
        "requested_num_patches": num_patches,
        "actual_num_patches": actual_num_patches,
        "batch_size": batch_size,
        "seed": seed,
        "repeats": repeats,
        "threshold": threshold if foreground_value is None else None,
        "foreground_value": foreground_value,
        "cpu": cpu_result,
        "gpu": gpu_result,
        "speedup_median": speedup,
        "mean_cldice_absolute_difference": statistics.mean(cldice_differences),
        "max_cldice_absolute_difference": max(cldice_differences),
        "prediction_skeleton_total_different_pixels": sum(
            int(item["different_pixels"]) for item in prediction_similarities
        ),
        "target_skeleton_total_different_pixels": sum(
            int(item["different_pixels"]) for item in target_similarities
        ),
        "minimum_prediction_skeleton_dice": min(
            float(item["dice"]) for item in prediction_similarities
        ),
        "minimum_target_skeleton_dice": min(
            float(item["dice"]) for item in target_similarities
        ),
        "artifacts": artifacts,
    }
    save_json(output_dir / "summary.json", summary)
    print(
        f"CPU={cpu_result['median_seconds']:.3f}s "
        f"({cpu_result['median_patches_per_second']:.2f} patches/s) | "
        f"GPU={gpu_result['median_seconds']:.3f}s "
        f"({gpu_result['median_patches_per_second']:.2f} patches/s) | "
        f"speedup={speedup:.2f}x",
        flush=True,
    )
    print(
        f"max clDice difference={max(cldice_differences):.3e} | "
        "CPU/GPU differing skeleton pixels="
        f"{summary['prediction_skeleton_total_different_pixels'] + summary['target_skeleton_total_different_pixels']}",
        flush=True,
    )
    print(f"Artifacts written to {output_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    run_patch_benchmark(
        args.prediction_mask,
        args.target_mask,
        args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        seed=args.seed,
        threshold=args.threshold,
        foreground_value=args.foreground_value,
        cuda_device=args.cuda_device,
        repeats=args.repeats,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
