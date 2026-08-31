from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize

from src.metrics.segmentation import (
    cldice_score_from_skeletons,
)
from src.utils.io import ensure_dir, save_csv, save_json, save_mask_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark paper-reference hard clDice on CPU and CUDA and compare "
            "the resulting skeletons."
        )
    )
    parser.add_argument(
        "--prediction-mask",
        required=True,
        help="Binary prediction mask, or a class-index mask with --foreground-value.",
    )
    parser.add_argument(
        "--target-mask",
        required=True,
        help="Binary target mask, or a class-index mask with --foreground-value.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cldice-benchmark",
        help="Directory for JSON/CSV results, skeletons, and overlap images.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Pixels greater than this value are foreground for binary masks.",
    )
    parser.add_argument(
        "--foreground-value",
        type=int,
        default=None,
        help="Select exactly this class value instead of thresholding both masks.",
    )
    parser.add_argument(
        "--cuda-device",
        default="cuda:0",
        help="CUDA device used for the GPU benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured repetitions per device; full-resolution CPU runs are expensive.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="clDice smoothing value, matching the production hard metric default.",
    )
    parser.add_argument(
        "--skeletonizer",
        choices=("paper",),
        default="paper",
        help=(
            "paper compares scikit-image Zhang thinning on CPU with equivalent "
            "PyTorch Zhang-Suen thinning on CUDA."
        ),
    )
    return parser.parse_args()


def load_binary_mask(
    path: str | Path,
    *,
    threshold: int,
    foreground_value: int | None,
) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8)
    if foreground_value is not None:
        return array == foreground_value
    return array > threshold


def skeleton_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(
            f"Skeleton shapes differ: first={first.shape}, second={second.shape}."
        )
    intersection = int(np.count_nonzero(first & second))
    first_pixels = int(np.count_nonzero(first))
    second_pixels = int(np.count_nonzero(second))
    union = first_pixels + second_pixels - intersection
    denominator = first_pixels + second_pixels
    return {
        "first_pixels": first_pixels,
        "second_pixels": second_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "different_pixels": int(np.count_nonzero(first ^ second)),
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "iou": 1.0 if union == 0 else intersection / union,
    }


def create_overlap_image(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """White=overlap, red=first only, cyan=second only, black=neither."""
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(
            f"Skeleton shapes differ: first={first.shape}, second={second.shape}."
        )
    output = np.zeros((*first.shape, 3), dtype=np.uint8)
    output[first & ~second] = (255, 0, 0)
    output[~first & second] = (0, 255, 255)
    output[first & second] = (255, 255, 255)
    return output


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_numpy_batch(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        return mask[None, ...], True
    if mask.ndim == 3:
        return mask, False
    raise ValueError(f"Expected a 2D or 3D mask array, got shape {mask.shape}.")


def skimage_skeletonize_masks(mask: np.ndarray) -> np.ndarray:
    """Skeletonize independent 2D masks with the paper's reference algorithm."""
    batch, squeeze = _as_numpy_batch(mask)
    skeletons = np.stack(
        [skeletonize(sample, method="zhang") for sample in batch],
        axis=0,
    )
    return skeletons[0] if squeeze else skeletons


_ZHANG_LUT_VALUES = (
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
)


def _zhang_suen_subiteration(
    mask: torch.Tensor,
    lut: torch.Tensor,
    *,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = F.pad(mask, (1, 1, 1, 1), value=False)
    north = padded[..., :-2, 1:-1]
    northeast = padded[..., :-2, 2:]
    east = padded[..., 1:-1, 2:]
    southeast = padded[..., 2:, 2:]
    south = padded[..., 2:, 1:-1]
    southwest = padded[..., 2:, :-2]
    west = padded[..., 1:-1, :-2]
    northwest = padded[..., :-2, :-2]
    neighborhood_index = (
        northwest.to(torch.int64)
        + 2 * north
        + 4 * northeast
        + 8 * east
        + 16 * southeast
        + 32 * south
        + 64 * southwest
        + 128 * west
    )
    labels = lut[neighborhood_index]
    phase = 1 if first else 2
    remove = mask & ((labels == 3) | (labels == phase))
    return mask & ~remove, remove


def torch_zhang_skeletonize_masks(mask: torch.Tensor) -> torch.Tensor:
    """Topology-preserving batched 2D Zhang-Suen thinning for CPU or CUDA."""
    original_ndim = mask.ndim
    if original_ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        mask = mask.unsqueeze(1)
    elif original_ndim != 4:
        raise ValueError(
            f"Expected a 2D, 3D, or 4D mask tensor, got shape {tuple(mask.shape)}."
        )
    thinned = mask > 0.5
    lut = torch.tensor(
        _ZHANG_LUT_VALUES,
        dtype=torch.uint8,
        device=thinned.device,
    )
    max_iterations = max(thinned.shape[-2:])
    for _ in range(max_iterations):
        thinned, removed_first = _zhang_suen_subiteration(
            thinned, lut, first=True
        )
        thinned, removed_second = _zhang_suen_subiteration(
            thinned, lut, first=False
        )
        if not bool(removed_first.any() | removed_second.any()):
            break
    if original_ndim == 2:
        return thinned[0, 0]
    if original_ndim == 3:
        return thinned[:, 0]
    return thinned


def _numpy_cldice_from_skeletons(
    prediction_skeleton: np.ndarray,
    target_skeleton: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float,
) -> float:
    prediction_skeletons, _ = _as_numpy_batch(prediction_skeleton)
    target_skeletons, _ = _as_numpy_batch(target_skeleton)
    predictions, _ = _as_numpy_batch(prediction)
    targets, _ = _as_numpy_batch(target)
    scores = []
    for pred_skel, target_skel, pred, truth in zip(
        prediction_skeletons,
        target_skeletons,
        predictions,
        targets,
    ):
        topology_precision = (
            np.count_nonzero(pred_skel & truth) + smooth
        ) / (np.count_nonzero(pred_skel) + smooth)
        topology_sensitivity = (
            np.count_nonzero(target_skel & pred) + smooth
        ) / (np.count_nonzero(target_skel) + smooth)
        denominator = topology_precision + topology_sensitivity
        scores.append(
            0.0
            if denominator == 0.0
            else 2.0
            * topology_precision
            * topology_sensitivity
            / denominator
        )
    return float(statistics.mean(scores))


def benchmark_device(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    device: torch.device,
    repeats: int,
    smooth: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if device.type == "cpu":
        return benchmark_paper_cpu(
            prediction_mask,
            target_mask,
            repeats=repeats,
            smooth=smooth,
        )
    return benchmark_paper_gpu(
        prediction_mask,
        target_mask,
        device=device,
        repeats=repeats,
        smooth=smooth,
    )


def benchmark_paper_cpu(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    repeats: int,
    smooth: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    skimage_skeletonize_masks(np.zeros((16, 16), dtype=bool))
    timings = []
    score = 0.0
    prediction_skeleton = target_skeleton = None
    for repeat in range(1, repeats + 1):
        print(f"  cpu skimage repeat {repeat}/{repeats}...", flush=True)
        started = time.perf_counter()
        prediction_skeleton = skimage_skeletonize_masks(prediction_mask)
        target_skeleton = skimage_skeletonize_masks(target_mask)
        score = _numpy_cldice_from_skeletons(
            prediction_skeleton,
            target_skeleton,
            prediction_mask,
            target_mask,
            smooth,
        )
        timings.append(time.perf_counter() - started)
    assert prediction_skeleton is not None and target_skeleton is not None
    return {
        "device": "cpu",
        "algorithm": "skimage.morphology.skeletonize(method=zhang)",
        "cldice": score,
        "transfer_seconds": 0.0,
        "timings_seconds": timings,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }, prediction_skeleton, target_skeleton


def benchmark_paper_gpu(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    device: torch.device,
    repeats: int,
    smooth: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for the GPU Zhang benchmark.")
    sample = torch.zeros((16, 16), dtype=torch.bool, device=device)
    torch_zhang_skeletonize_masks(sample)
    _synchronize(device)
    transfer_started = time.perf_counter()
    prediction = torch.from_numpy(np.ascontiguousarray(prediction_mask)).to(device)
    target = torch.from_numpy(np.ascontiguousarray(target_mask)).to(device)
    _synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_started
    baseline_bytes = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    score = 0.0
    prediction_skeleton = target_skeleton = None
    with torch.inference_mode():
        for repeat in range(1, repeats + 1):
            print(f"  {device} Zhang-Suen repeat {repeat}/{repeats}...", flush=True)
            _synchronize(device)
            started = time.perf_counter()
            prediction_skeleton = torch_zhang_skeletonize_masks(prediction)
            target_skeleton = torch_zhang_skeletonize_masks(target)
            score = cldice_score_from_skeletons(
                prediction_skeleton,
                target_skeleton,
                prediction,
                target,
                smooth=smooth,
            )
            _synchronize(device)
            timings.append(time.perf_counter() - started)
    assert prediction_skeleton is not None and target_skeleton is not None
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "device": str(device),
        "algorithm": "PyTorch Zhang-Suen thinning",
        "cldice": score,
        "transfer_seconds": transfer_seconds,
        "timings_seconds": timings,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "cuda_name": torch.cuda.get_device_name(device),
        "baseline_cuda_allocated_bytes": baseline_bytes,
        "peak_cuda_allocated_bytes": peak_bytes,
        "peak_extra_cuda_allocated_bytes": peak_bytes - baseline_bytes,
    }
    return (
        result,
        prediction_skeleton.detach().cpu().numpy().astype(bool),
        target_skeleton.detach().cpu().numpy().astype(bool),
    )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def run_benchmark(
    prediction_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: int = 127,
    foreground_value: int | None = None,
    cuda_device: str = "cuda:0",
    repeats: int = 1,
    smooth: float = 1e-6,
    skeletonizer: str = "paper",
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
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction and target mask dimensions differ: "
            f"prediction={prediction_mask.shape}, target={target_mask.shape}."
        )

    print(
        f"Mask shape: {prediction_mask.shape[1]}x{prediction_mask.shape[0]} | "
        f"prediction foreground={int(prediction_mask.sum())} | "
        f"target foreground={int(target_mask.sum())}",
        flush=True,
    )
    cuda = torch.device(cuda_device)
    if cuda.type != "cuda":
        raise ValueError("--cuda-device must name a CUDA device, such as cuda:0.")
    if skeletonizer != "paper":
        raise ValueError("skeletonizer must be 'paper'.")
    print("Benchmarking paper-reference CPU skeletonization...", flush=True)
    cpu_result, prediction_cpu, target_cpu = benchmark_paper_cpu(
        prediction_mask,
        target_mask,
        repeats=repeats,
        smooth=smooth,
    )
    print(f"Benchmarking {cuda}...", flush=True)
    try:
        gpu_result, prediction_gpu, target_gpu = benchmark_paper_gpu(
            prediction_mask,
            target_mask,
            device=cuda,
            repeats=repeats,
            smooth=smooth,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA ran out of memory during full-resolution hard clDice. "
            "Stop competing GPU jobs or benchmark a smaller mask."
        ) from error

    prediction_similarity = skeleton_similarity(prediction_cpu, prediction_gpu)
    target_similarity = skeleton_similarity(target_cpu, target_gpu)
    speedup = cpu_result["median_seconds"] / gpu_result["median_seconds"]
    cldice_difference = abs(cpu_result["cldice"] - gpu_result["cldice"])

    artifacts = {
        "prediction_mask": "prediction_mask.png",
        "target_mask": "target_mask.png",
        "prediction_cpu_skeleton": "prediction_cpu_skeleton.png",
        "prediction_gpu_skeleton": "prediction_gpu_skeleton.png",
        "target_cpu_skeleton": "target_cpu_skeleton.png",
        "target_gpu_skeleton": "target_gpu_skeleton.png",
        "prediction_cpu_gpu_overlap": "prediction_cpu_vs_gpu_skeleton.png",
        "target_cpu_gpu_overlap": "target_cpu_vs_gpu_skeleton.png",
        "cpu_prediction_target_overlap": "cpu_prediction_vs_target_skeleton.png",
        "gpu_prediction_target_overlap": "gpu_prediction_vs_target_skeleton.png",
    }
    save_mask_image(output_dir / artifacts["prediction_mask"], prediction_mask * 255)
    save_mask_image(output_dir / artifacts["target_mask"], target_mask * 255)
    save_mask_image(
        output_dir / artifacts["prediction_cpu_skeleton"], prediction_cpu * 255
    )
    save_mask_image(
        output_dir / artifacts["prediction_gpu_skeleton"], prediction_gpu * 255
    )
    save_mask_image(output_dir / artifacts["target_cpu_skeleton"], target_cpu * 255)
    save_mask_image(output_dir / artifacts["target_gpu_skeleton"], target_gpu * 255)
    _save_rgb(
        output_dir / artifacts["prediction_cpu_gpu_overlap"],
        create_overlap_image(prediction_cpu, prediction_gpu),
    )
    _save_rgb(
        output_dir / artifacts["target_cpu_gpu_overlap"],
        create_overlap_image(target_cpu, target_gpu),
    )
    _save_rgb(
        output_dir / artifacts["cpu_prediction_target_overlap"],
        create_overlap_image(prediction_cpu, target_cpu),
    )
    _save_rgb(
        output_dir / artifacts["gpu_prediction_target_overlap"],
        create_overlap_image(prediction_gpu, target_gpu),
    )

    summary = {
        "prediction_mask": str(prediction_path),
        "target_mask": str(target_path),
        "output_dir": str(output_dir),
        "width": int(prediction_mask.shape[1]),
        "height": int(prediction_mask.shape[0]),
        "prediction_foreground_pixels": int(prediction_mask.sum()),
        "target_foreground_pixels": int(target_mask.sum()),
        "threshold": threshold if foreground_value is None else None,
        "foreground_value": foreground_value,
        "repeats": repeats,
        "smooth": smooth,
        "skeletonizer": skeletonizer,
        "cpu": cpu_result,
        "gpu": gpu_result,
        "speedup_median": speedup,
        "cldice_absolute_difference": cldice_difference,
        "prediction_skeleton_cpu_gpu": prediction_similarity,
        "target_skeleton_cpu_gpu": target_similarity,
        "overlap_legend": {
            "white": "present in both skeletons",
            "red": "present only in the first named skeleton",
            "cyan": "present only in the second named skeleton",
            "black": "present in neither skeleton",
        },
        "artifacts": artifacts,
    }
    save_json(output_dir / "summary.json", summary)
    save_csv(
        output_dir / "timings.csv",
        [
            {
                "device": result["device"],
                "algorithm": result["algorithm"],
                "cldice": result["cldice"],
                "transfer_seconds": result["transfer_seconds"],
                "mean_seconds": result["mean_seconds"],
                "median_seconds": result["median_seconds"],
                "min_seconds": result["min_seconds"],
                "max_seconds": result["max_seconds"],
                "speedup_vs_gpu": speedup if result is cpu_result else 1.0,
            }
            for result in (cpu_result, gpu_result)
        ],
    )

    print(
        f"CPU clDice={cpu_result['cldice']:.10f} | "
        f"median={cpu_result['median_seconds']:.3f}s",
        flush=True,
    )
    print(
        f"GPU clDice={gpu_result['cldice']:.10f} | "
        f"median={gpu_result['median_seconds']:.3f}s | speedup={speedup:.2f}x",
        flush=True,
    )
    print(
        f"clDice absolute difference={cldice_difference:.3e} | "
        f"prediction skeleton Dice={prediction_similarity['dice']:.10f} | "
        f"target skeleton Dice={target_similarity['dice']:.10f}",
        flush=True,
    )
    print(f"Artifacts written to {output_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    run_benchmark(
        args.prediction_mask,
        args.target_mask,
        args.output_dir,
        threshold=args.threshold,
        foreground_value=args.foreground_value,
        cuda_device=args.cuda_device,
        repeats=args.repeats,
        smooth=args.smooth,
        skeletonizer=args.skeletonizer,
    )


if __name__ == "__main__":
    main()
