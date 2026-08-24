from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OriginalImageRecord:
    source_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    mask_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class PatchRecord:
    source_id: str
    image_path: Path
    mask_path: Path
    x: int
    y: int
    patch_size: int
    scale: float = 1.0
    scaled_width: int = 0
    scaled_height: int = 0
    resolution_bucket: str = "normal"
    scale_label: str = "normal"
    source_crop_size: int | None = None
    mask_paths: dict[str, Path] | None = None


def build_original_image_records(pairs: Iterable[tuple[Path, Path | dict[str, Path]]]) -> list[OriginalImageRecord]:
    records: list[OriginalImageRecord] = []
    for image_path, mask_value in pairs:
        mask_paths = dict(mask_value) if isinstance(mask_value, dict) else None
        mask_path = (mask_paths.get("loci") or next(iter(mask_paths.values()))) if mask_paths else mask_value
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            OriginalImageRecord(
                source_id=image_path.name,
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
                mask_paths=mask_paths,
            )
        )
    return records


def _compute_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]

    positions = list(range(0, max(length - patch_size, 0) + 1, stride))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def compute_shifted_positions(length: int, patch_size: int, stride: int, offset: int) -> list[int]:
    if length <= patch_size:
        return [0]

    final_position = length - patch_size
    bounded_offset = max(0, min(int(offset), final_position))
    positions = {0, final_position}
    positions.update(range(bounded_offset, final_position + 1, stride))
    return sorted(positions)


def _count_foreground(mask_patch: np.ndarray, mask_threshold: int) -> int:
    if mask_patch.ndim == 3:
        mask_patch = mask_patch[..., 0]
    return int((mask_patch > mask_threshold).sum())


def _resampling_filter(name: str, *, is_mask: bool = False) -> Image.Resampling:
    normalized = str(name).strip().lower()
    if is_mask and normalized == "foreground_preserving":
        return Image.Resampling.BOX
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if normalized not in filters:
        raise ValueError(f"Unsupported resampling filter: {name}")
    return filters[normalized]


def crop_and_pad_array(array: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    cropped = array[y : y + patch_size, x : x + patch_size]
    height, width = cropped.shape[:2]
    if height == patch_size and width == patch_size:
        return cropped

    if array.ndim == 3:
        padded = np.zeros((patch_size, patch_size, array.shape[2]), dtype=array.dtype)
    else:
        padded = np.zeros((patch_size, patch_size), dtype=array.dtype)
    padded[:height, :width] = cropped
    return padded


def _centered_source_bounds(
    width: int,
    height: int,
    x: int,
    y: int,
    patch_size: int,
    scale: float,
) -> tuple[int, int, int, int]:
    crop_size = max(patch_size, int(round(patch_size * float(scale))))
    center_x = x + (patch_size / 2.0)
    center_y = y + (patch_size / 2.0)
    x0 = int(round(center_x - (crop_size / 2.0)))
    y0 = int(round(center_y - (crop_size / 2.0)))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height

    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def crop_scaled_image_patch(
    array: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    scale: float,
    resampling: str = "lanczos",
) -> np.ndarray:
    if math.isclose(scale, 1.0):
        return crop_and_pad_array(array, x, y, patch_size)

    height, width = array.shape[:2]
    source_x0, source_y0, source_x1, source_y1 = _centered_source_bounds(
        width,
        height,
        x,
        y,
        patch_size,
        scale,
    )
    cropped = array[source_y0:source_y1, source_x0:source_x1]
    if cropped.size == 0:
        return np.zeros((patch_size, patch_size, array.shape[2]), dtype=array.dtype)

    image = Image.fromarray(cropped)
    resized = image.resize(
        (patch_size, patch_size),
        resample=_resampling_filter(resampling),
    )
    return np.array(resized, dtype=array.dtype)


def crop_scaled_mask_patch(
    array: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    scale: float,
    mask_threshold: int,
    resampling: str = "foreground_preserving",
) -> np.ndarray:
    if math.isclose(scale, 1.0):
        return crop_and_pad_array(array, x, y, patch_size)

    height, width = array.shape[:2]
    source_x0, source_y0, source_x1, source_y1 = _centered_source_bounds(
        width,
        height,
        x,
        y,
        patch_size,
        scale,
    )
    cropped = array[source_y0:source_y1, source_x0:source_x1]
    if cropped.size == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    binary = (cropped > mask_threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary)
    resized = mask_image.resize(
        (patch_size, patch_size),
        resample=_resampling_filter(resampling, is_mask=True),
    )
    resized_array = np.array(resized, dtype=np.uint8)
    if str(resampling).strip().lower() == "foreground_preserving":
        return (resized_array > 0).astype(np.uint8) * 255
    return resized_array


def _phase_config(patching_config: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_name = "train" if phase == "train" else "validation"
    return patching_config.get(phase_name, {})


def _epoch_offsets(
    patch_size: int,
    stride: int,
    phase_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[int, int]:
    offset_config = phase_config.get("random_offset", {})
    if not bool(offset_config.get("enabled", False)):
        return 0, 0

    max_fraction = float(offset_config.get("max_fraction_of_patch", 0.5))
    max_offset = min(max(stride - 1, 0), max(0, int(round(patch_size * max_fraction))))
    if max_offset <= 0:
        return 0, 0
    return int(rng.integers(0, max_offset + 1)), int(rng.integers(0, max_offset + 1))


def _valid_max_scale(
    width: int,
    height: int,
    x: int,
    y: int,
    patch_size: int,
    configured_max_scale: float,
) -> float:
    center_x = x + (patch_size / 2.0)
    center_y = y + (patch_size / 2.0)
    half_extent = min(center_x, center_y, width - center_x, height - center_y)
    if half_extent <= 0:
        return 1.0
    return max(1.0, min(float(configured_max_scale), (2.0 * half_extent) / patch_size))


def _sample_context_scale(
    width: int,
    height: int,
    x: int,
    y: int,
    patch_size: int,
    phase_config: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    scale_config = phase_config.get("scaled_context", {})
    if not bool(scale_config.get("enabled", False)):
        return 1.0

    probability = float(scale_config.get("probability", 0.25))
    if probability <= 0.0 or float(rng.random()) >= probability:
        return 1.0

    configured_max_scale = float(scale_config.get("max_scale", 2.0))
    max_scale = _valid_max_scale(width, height, x, y, patch_size, configured_max_scale)
    if max_scale <= 1.0:
        return 1.0

    beta_alpha = float(scale_config.get("beta_alpha", 1.0))
    beta_beta = float(scale_config.get("beta_beta", 4.0))
    t = float(rng.beta(beta_alpha, beta_beta))
    scale = 1.0 + (t * (max_scale - 1.0))
    crop_size = int(round(patch_size * scale))
    if crop_size <= patch_size:
        return 1.0
    return min(scale, max_scale)


def _source_crop_bounds(record: PatchRecord) -> tuple[int, int, int, int]:
    return _centered_source_bounds(
        record.scaled_width,
        record.scaled_height,
        record.x,
        record.y,
        record.patch_size,
        record.scale,
    )


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _filter_contained_patch_records(
    records: list[PatchRecord],
    phase_config: dict[str, Any],
) -> list[PatchRecord]:
    containment_config = (
        phase_config.get("scaled_context", {}).get("containment_filter", {})
    )
    if not bool(containment_config.get("enabled", False)):
        return records

    threshold = float(containment_config.get("threshold", 0.8))
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "scaled_context.containment_filter.threshold must be in (0, 1]."
        )
    if len(records) < 2:
        return records
    preserve_normal = bool(containment_config.get("preserve_normal_patches", True))

    bounds = [_source_crop_bounds(record) for record in records]
    areas = [
        max(0, x1 - x0) * max(0, y1 - y0)
        for x0, y0, x1, y1 in bounds
    ]
    largest_first = sorted(
        range(len(records)), key=lambda index: (-areas[index], index)
    )
    retained_larger_indices: list[int] = []
    removed_indices: set[int] = set()

    for index in largest_first:
        record = records[index]
        if preserve_normal and record.scale_label == "normal":
            retained_larger_indices.append(index)
            continue

        smaller_area = areas[index]
        if smaller_area > 0:
            for larger_index in retained_larger_indices:
                if areas[larger_index] <= smaller_area:
                    break
                covered_fraction = _intersection_area(
                    bounds[index], bounds[larger_index]
                ) / smaller_area
                if covered_fraction >= threshold:
                    removed_indices.add(index)
                    break

        if index not in removed_indices:
            retained_larger_indices.append(index)

    return [record for index, record in enumerate(records) if index not in removed_indices]


def build_patch_records(
    original_records: Iterable[OriginalImageRecord],
    patching_config: dict[str, Any],
    *,
    phase: str = "validation",
    epoch: int = 0,
    base_seed: int = 0,
) -> list[PatchRecord]:
    patch_size = int(patching_config["patch_size"])
    stride = int(patching_config["stride"])
    filter_empty_patches = bool(patching_config.get("filter_empty_patches", True))
    mask_threshold = int(patching_config.get("mask_threshold", 127))
    min_foreground_pixels = int(patching_config.get("min_foreground_pixels", 1))
    mask_resampling = str(patching_config.get("mask_resampling", "foreground_preserving"))
    phase_cfg = _phase_config(patching_config, phase)
    rng = np.random.default_rng(int(base_seed) + int(epoch))
    offset_x, offset_y = _epoch_offsets(patch_size, stride, phase_cfg, rng)

    patch_records: list[PatchRecord] = []
    for record in original_records:
        source_patch_records: list[PatchRecord] = []
        mask_arrays: list[np.ndarray] = []
        if record.mask_paths:
            mask_names = [
                name for name in ("loci", "inoculum") if name in record.mask_paths
            ]
            if not mask_names:
                mask_names = list(record.mask_paths)
            if (
                patching_config.get("include_join_masks", False)
                and "join" in record.mask_paths
            ):
                mask_names.append("join")
            paths = [record.mask_paths[name] for name in mask_names]
        else:
            paths = [record.mask_path]
        for mask_path in paths:
            with Image.open(mask_path) as mask_image:
                mask_arrays.append(np.array(mask_image.convert("L"), dtype=np.uint8))

        xs = compute_shifted_positions(record.width, patch_size, stride, offset_x)
        ys = compute_shifted_positions(record.height, patch_size, stride, offset_y)
        for y in ys:
            for x in xs:
                scale = _sample_context_scale(
                    record.width,
                    record.height,
                    x,
                    y,
                    patch_size,
                    phase_cfg,
                    rng,
                )
                mask_patches = [
                    crop_scaled_mask_patch(
                        mask_array, x=x, y=y, patch_size=patch_size, scale=scale,
                        mask_threshold=mask_threshold, resampling=mask_resampling,
                    ) for mask_array in mask_arrays
                ]
                foreground_pixels = int(np.logical_or.reduce(
                    [patch > mask_threshold for patch in mask_patches]
                ).sum())
                if filter_empty_patches and foreground_pixels < min_foreground_pixels:
                    continue

                scale_label = "scaled_context" if scale > 1.0 else "normal"
                source_patch_records.append(
                    PatchRecord(
                        source_id=record.source_id,
                        image_path=record.image_path,
                        mask_path=record.mask_path,
                        x=x,
                        y=y,
                        patch_size=patch_size,
                        scale=scale,
                        scaled_width=record.width,
                        scaled_height=record.height,
                        resolution_bucket=scale_label,
                        scale_label=scale_label,
                        source_crop_size=int(round(patch_size * scale)),
                        mask_paths=record.mask_paths,
                    )
                )

        patch_records.extend(
            _filter_contained_patch_records(source_patch_records, phase_cfg)
        )

    return patch_records


def build_legacy_patch_records(
    original_records: Iterable[OriginalImageRecord],
    patch_size: int,
    stride: int,
    filter_empty_patches: bool,
    mask_threshold: int,
    min_foreground_pixels: int,
) -> list[PatchRecord]:
    patching_config = {
        "patch_size": patch_size,
        "stride": stride,
        "filter_empty_patches": filter_empty_patches,
        "mask_threshold": mask_threshold,
        "min_foreground_pixels": min_foreground_pixels,
        "mask_resampling": "foreground_preserving",
        "validation": {
            "random_offset": {"enabled": False},
            "scaled_context": {"enabled": False},
        },
    }
    return build_patch_records(original_records, patching_config, phase="validation")
