from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def build_original_image_records(pairs: Iterable[tuple[Path, Path]]) -> list[OriginalImageRecord]:
    records: list[OriginalImageRecord] = []
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            OriginalImageRecord(
                source_id=image_path.stem,
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


def build_patch_records(
    original_records: Iterable[OriginalImageRecord],
    patch_size: int,
    stride: int,
    filter_empty_patches: bool,
    mask_threshold: int,
    min_foreground_pixels: int,
) -> list[PatchRecord]:
    patch_records: list[PatchRecord] = []

    for record in original_records:
        xs = _compute_positions(record.width, patch_size, stride)
        ys = _compute_positions(record.height, patch_size, stride)

        with Image.open(record.mask_path) as mask_image:
            mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)

        for y in ys:
            for x in xs:
                mask_patch = crop_and_pad_array(mask_array, x, y, patch_size)
                if filter_empty_patches:
                    foreground_pixels = _count_foreground(mask_patch, mask_threshold)
                    if foreground_pixels < min_foreground_pixels:
                        continue

                patch_records.append(
                    PatchRecord(
                        source_id=record.source_id,
                        image_path=record.image_path,
                        mask_path=record.mask_path,
                        x=x,
                        y=y,
                        patch_size=patch_size,
                    )
                )

    return patch_records
