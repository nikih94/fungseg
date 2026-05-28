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
    resolution_bucket: str = "native"
    scale_label: str = "native"


def build_original_image_records(pairs: Iterable[tuple[Path, Path]]) -> list[OriginalImageRecord]:
    records: list[OriginalImageRecord] = []
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            OriginalImageRecord(
                source_id=image_path.name,
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
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


def _scaled_dimension(length: int, scale: float) -> int:
    return max(1, int(round(length * scale)))


def _resolution_bucket(scaled_width: int, scaled_height: int, scale_label: str) -> str:
    long_edge = max(scaled_width, scaled_height)
    for bucket_edge in (1200, 1600, 2400, 3200):
        if long_edge <= bucket_edge:
            return f"bucket_{bucket_edge}"
    return "native_large" if scale_label == "native" else "bucket_native_large"


def resolve_scale_specs(
    record: OriginalImageRecord,
    multiscale_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = multiscale_config or {}
    if not bool(config.get("enabled", False)):
        return [
            {
                "scale": 1.0,
                "scaled_width": record.width,
                "scaled_height": record.height,
                "scale_label": "native",
                "resolution_bucket": _resolution_bucket(record.width, record.height, "native"),
            }
        ]

    include_native = bool(config.get("include_native", True))
    target_long_edges = [int(value) for value in config.get("target_long_edges", [])]
    max_scale = float(config.get("max_scale", 1.0))
    deduplicate_tolerance = float(config.get("deduplicate_scale_tolerance", 0.03))
    native_long_edge = max(record.width, record.height)

    specs: list[dict[str, Any]] = []

    def add_spec(scale: float, scale_label: str) -> None:
        scale = min(float(scale), max_scale)
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}.")
        duplicate_indices = [
            index
            for index, existing in enumerate(specs)
            if abs(float(existing["scale"]) - scale) <= deduplicate_tolerance
        ]
        if duplicate_indices and scale_label != "native":
            return
        for index in reversed(duplicate_indices):
            specs.pop(index)
        for existing in specs:
            if abs(float(existing["scale"]) - scale) <= deduplicate_tolerance:
                return
        scaled_width = _scaled_dimension(record.width, scale)
        scaled_height = _scaled_dimension(record.height, scale)
        specs.append(
            {
                "scale": scale,
                "scaled_width": scaled_width,
                "scaled_height": scaled_height,
                "scale_label": scale_label,
                "resolution_bucket": _resolution_bucket(scaled_width, scaled_height, scale_label),
            }
        )

    for target_long_edge in sorted(target_long_edges):
        scale = target_long_edge / native_long_edge
        if scale < max_scale:
            add_spec(scale, f"long_edge_{target_long_edge}")

    if include_native:
        add_spec(1.0, "native")

    if not specs:
        add_spec(1.0, "native")

    return specs


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

    source_x0 = max(0, int(math.floor(x / scale)))
    source_y0 = max(0, int(math.floor(y / scale)))
    source_x1 = min(array.shape[1], int(math.ceil((x + patch_size) / scale)))
    source_y1 = min(array.shape[0], int(math.ceil((y + patch_size) / scale)))
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

    source_x0 = max(0, int(math.floor(x / scale)))
    source_y0 = max(0, int(math.floor(y / scale)))
    source_x1 = min(array.shape[1], int(math.ceil((x + patch_size) / scale)))
    source_y1 = min(array.shape[0], int(math.ceil((y + patch_size) / scale)))
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


def build_patch_records(
    original_records: Iterable[OriginalImageRecord],
    patch_size: int,
    stride: int,
    filter_empty_patches: bool,
    mask_threshold: int,
    min_foreground_pixels: int,
    multiscale_config: dict[str, Any] | None = None,
) -> list[PatchRecord]:
    patch_records: list[PatchRecord] = []
    mask_resampling = (multiscale_config or {}).get("mask_resampling", "foreground_preserving")

    for record in original_records:
        with Image.open(record.mask_path) as mask_image:
            mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)

        for scale_spec in resolve_scale_specs(record, multiscale_config):
            scaled_width = int(scale_spec["scaled_width"])
            scaled_height = int(scale_spec["scaled_height"])
            xs = _compute_positions(scaled_width, patch_size, stride)
            ys = _compute_positions(scaled_height, patch_size, stride)

            for y in ys:
                for x in xs:
                    mask_patch = crop_scaled_mask_patch(
                        mask_array,
                        x=x,
                        y=y,
                        patch_size=patch_size,
                        scale=float(scale_spec["scale"]),
                        mask_threshold=mask_threshold,
                        resampling=str(mask_resampling),
                    )
                    foreground_pixels = _count_foreground(mask_patch, mask_threshold)
                    if filter_empty_patches and foreground_pixels < min_foreground_pixels:
                        continue

                    patch_records.append(
                        PatchRecord(
                            source_id=record.source_id,
                            image_path=record.image_path,
                            mask_path=record.mask_path,
                            x=x,
                            y=y,
                            patch_size=patch_size,
                            scale=float(scale_spec["scale"]),
                            scaled_width=scaled_width,
                            scaled_height=scaled_height,
                            resolution_bucket=str(scale_spec["resolution_bucket"]),
                            scale_label=str(scale_spec["scale_label"]),
                        )
                    )

    return patch_records
