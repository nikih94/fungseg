from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import albumentations as A
import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import compose_multiclass_mask
from src.data.soft_cldice_iterations import required_soft_skeleton_iterations
from src.patching import OriginalImageRecord, _compute_positions, crop_and_pad_array


@dataclass(frozen=True)
class StaticPatchRecord:
    """One immutable, larger-than-model-input training cache region."""

    cache_index: int
    source_id: str
    image_path: Path
    mask_path: Path
    anchor_x: int
    anchor_y: int
    cache_x: int
    cache_y: int
    patch_size: int
    cache_size: int
    source_width: int
    source_height: int
    mask_paths: dict[str, Path] | None = None
    soft_cldice_iterations: int | None = None


@dataclass(frozen=True)
class CachedTrainingCropRecord:
    """Epoch-specific model crop selected from a static cache region."""

    cache_index: int
    source_id: str
    x: int
    y: int
    local_x: int
    local_y: int
    patch_size: int
    foreground_pixels: int
    is_background_only: bool
    soft_cldice_iterations: int | None = None
    scale: float = 1.0
    scaled_width: int = 0
    scaled_height: int = 0
    resolution_bucket: str = "normal"
    scale_label: str = "normal"
    source_crop_size: int | None = None


class StaticPatchCache:
    """Run-level cache metadata and ownership for static training regions."""

    def __init__(
        self,
        cache_dir: Path,
        records: list[StaticPatchRecord],
        mask_names: tuple[str, ...],
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.records = records
        self.mask_names = mask_names
        self._masks: np.ndarray | None = None

    def masks(self) -> np.ndarray:
        if self._masks is None:
            self._masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")
        return self._masks

    def close(self) -> None:
        self._masks = None

    def cleanup(self) -> None:
        self.close()
        shutil.rmtree(self.cache_dir.parent, ignore_errors=True)


def remove_stale_patch_caches(run_dir: str | Path) -> None:
    root = Path(run_dir) / ".train_patch_cache"
    if root.exists():
        shutil.rmtree(root)


def _cache_start(length: int, anchor: int, patch_size: int, overlap: int) -> int:
    cache_size = patch_size + overlap
    if length <= cache_size:
        return 0
    desired = anchor - (overlap // 2)
    return max(0, min(desired, length - cache_size))


def _mask_names(segmentation_mode: str, merge_join_masks: bool) -> tuple[str, ...]:
    if segmentation_mode == "multiclass":
        names = ["loci", "inoculum"]
        if merge_join_masks:
            names.append("join")
        return tuple(names)
    return ("binary",)


def _effective_loci_target(
    masks: np.ndarray,
    mask_names: tuple[str, ...],
    mask_threshold: int,
    segmentation_mode: str,
    merge_join_masks: bool,
) -> np.ndarray:
    by_name = {name: masks[index] for index, name in enumerate(mask_names)}
    if segmentation_mode != "multiclass":
        return by_name["binary"] > mask_threshold
    target, _ = compose_multiclass_mask(
        by_name["loci"],
        by_name["inoculum"],
        mask_threshold,
        join_mask=by_name.get("join"),
        merge_join_masks=merge_join_masks,
    )
    return target == 1


def _adjust_iterations(required: int, margin: int, round_up_to: int) -> int:
    adjusted = int(required) + int(margin)
    return ((adjusted + round_up_to - 1) // round_up_to) * round_up_to


def build_static_patch_cache(
    original_records: Iterable[OriginalImageRecord],
    run_dir: str | Path,
    patching_config: dict[str, Any],
    *,
    segmentation_mode: str,
    merge_join_masks: bool,
    compute_soft_cldice_iterations: bool,
    iteration_margin: int,
    iteration_round_up_to: int,
) -> StaticPatchCache:
    """Decode training sources once and cache larger static uint8 regions."""
    originals = list(original_records)
    patch_size = int(patching_config["patch_size"])
    overlap = int(patching_config["overlap"])
    stride = int(patching_config["stride"])
    cache_size = patch_size + overlap
    threshold = int(patching_config.get("mask_threshold", 127))
    mask_names = _mask_names(segmentation_mode, merge_join_masks)

    provisional: list[StaticPatchRecord] = []
    for original in originals:
        xs = _compute_positions(original.width, patch_size, stride)
        ys = _compute_positions(original.height, patch_size, stride)
        for anchor_y in ys:
            for anchor_x in xs:
                provisional.append(
                    StaticPatchRecord(
                        cache_index=len(provisional),
                        source_id=original.source_id,
                        image_path=original.image_path,
                        mask_path=original.mask_path,
                        anchor_x=anchor_x,
                        anchor_y=anchor_y,
                        cache_x=_cache_start(
                            original.width, anchor_x, patch_size, overlap
                        ),
                        cache_y=_cache_start(
                            original.height, anchor_y, patch_size, overlap
                        ),
                        patch_size=patch_size,
                        cache_size=cache_size,
                        source_width=original.width,
                        source_height=original.height,
                        mask_paths=original.mask_paths,
                    )
                )
    if not provisional:
        raise ValueError("Cannot build an empty static training patch cache.")

    root = Path(run_dir) / ".train_patch_cache"
    remove_stale_patch_caches(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    required_bytes = (
        len(provisional)
        * cache_size
        * cache_size
        * (3 + len(mask_names))
        + len(provisional) * np.dtype(np.int64).itemsize
    )
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < required_bytes * 1.1:
        raise OSError(
            "Insufficient free space for static training patch cache: need about "
            f"{required_bytes} bytes, have {free_bytes} bytes in {root}."
        )

    staging = root / f"staging-{uuid.uuid4().hex}"
    ready = root / "ready"
    staging.mkdir()
    completed_records: list[StaticPatchRecord] = []
    try:
        images = open_memmap(
            staging / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(provisional), cache_size, cache_size, 3),
        )
        masks = open_memmap(
            staging / "masks.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(provisional), len(mask_names), cache_size, cache_size),
        )
        iterations = open_memmap(
            staging / "soft_cldice_iterations.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(provisional),),
        )
        iterations[:] = -1

        by_source: dict[str, list[StaticPatchRecord]] = {}
        for record in provisional:
            by_source.setdefault(record.source_id, []).append(record)

        for source_records in by_source.values():
            first = source_records[0]
            with Image.open(first.image_path) as handle:
                source_image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            source_masks: dict[str, np.ndarray] = {}
            if segmentation_mode == "multiclass":
                if not first.mask_paths:
                    raise ValueError("Multiclass cache records require named masks.")
                for name in mask_names:
                    path = first.mask_paths.get(name)
                    if path is None:
                        source_masks[name] = np.zeros(
                            (first.source_height, first.source_width), dtype=np.uint8
                        )
                    else:
                        with Image.open(path) as handle:
                            source_masks[name] = np.asarray(
                                handle.convert("L"), dtype=np.uint8
                            )
            else:
                with Image.open(first.mask_path) as handle:
                    source_masks["binary"] = np.asarray(
                        handle.convert("L"), dtype=np.uint8
                    )

            for record in source_records:
                index = record.cache_index
                images[index] = crop_and_pad_array(
                    source_image, record.cache_x, record.cache_y, cache_size
                )
                cached_masks = np.stack(
                    [
                        crop_and_pad_array(
                            source_masks[name],
                            record.cache_x,
                            record.cache_y,
                            cache_size,
                        )
                        for name in mask_names
                    ]
                )
                masks[index] = cached_masks
                value: int | None = None
                if compute_soft_cldice_iterations:
                    required = required_soft_skeleton_iterations(
                        _effective_loci_target(
                            cached_masks,
                            mask_names,
                            threshold,
                            segmentation_mode,
                            merge_join_masks,
                        )
                    )
                    value = _adjust_iterations(
                        required, iteration_margin, iteration_round_up_to
                    )
                    iterations[index] = value
                completed_records.append(
                    replace(record, soft_cldice_iterations=value)
                )

        images.flush()
        masks.flush()
        iterations.flush()
        del images, masks, iterations
        (staging / "READY").touch()
        staging.rename(ready)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    completed_records.sort(key=lambda item: item.cache_index)
    return StaticPatchCache(ready, completed_records, mask_names)


def _seeded_rng(*parts: object) -> np.random.Generator:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _sample_axis(
    *,
    anchor: int,
    cache_start: int,
    source_length: int,
    patch_size: int,
    cache_size: int,
    max_motion: int,
    random_enabled: bool,
    rng: np.random.Generator,
) -> int:
    final = max(0, source_length - patch_size)
    if (
        not random_enabled
        or source_length <= patch_size
        or anchor == 0
        or anchor == final
        or max_motion <= 0
    ):
        return anchor
    available_end = min(final, cache_start + cache_size - patch_size)
    before = max_motion // 2
    after = max_motion - before
    low = max(cache_start, anchor - before)
    high = min(available_end, anchor + after)
    if high <= low:
        return low
    return int(rng.integers(low, high + 1))


def build_epoch_training_crop_records(
    cache: StaticPatchCache,
    source_ids: Iterable[str],
    patching_config: dict[str, Any],
    *,
    epoch: int,
    base_seed: int,
    fold_index: int,
    segmentation_mode: str,
    merge_join_masks: bool,
) -> list[CachedTrainingCropRecord]:
    """Select and foreground-filter one final crop per static region."""
    selected_sources = set(source_ids)
    patch_size = int(patching_config["patch_size"])
    overlap = int(patching_config["overlap"])
    stride = int(patching_config["stride"])
    threshold = int(patching_config.get("mask_threshold", 127))
    minimum = int(patching_config.get("min_foreground_pixels", 1))
    filter_empty = bool(patching_config.get("filter_empty_patches", True))
    train_config = patching_config.get("train", {})
    random_config = train_config.get("random_offset", {})
    random_enabled = bool(random_config.get("enabled", False))
    max_motion = min(
        overlap,
        max(stride - 1, 0),
        max(
            0,
            int(
                round(
                    patch_size
                    * float(random_config.get("max_fraction_of_patch", 0.5))
                )
            ),
        ),
    )
    background_config = train_config.get("background_only", {})
    background_enabled = bool(background_config.get("enabled", False))
    background_percentage = float(
        background_config.get("percentage_of_foreground", 0.0)
    )
    masks = cache.masks()
    by_name = {name: index for index, name in enumerate(cache.mask_names)}
    candidates_by_source: dict[str, list[CachedTrainingCropRecord]] = {}

    for static in cache.records:
        if static.source_id not in selected_sources:
            continue
        rng = _seeded_rng(
            base_seed, fold_index, epoch, static.source_id, static.cache_index
        )
        x = _sample_axis(
            anchor=static.anchor_x,
            cache_start=static.cache_x,
            source_length=static.source_width,
            patch_size=patch_size,
            cache_size=static.cache_size,
            max_motion=max_motion,
            random_enabled=random_enabled,
            rng=rng,
        )
        y = _sample_axis(
            anchor=static.anchor_y,
            cache_start=static.cache_y,
            source_length=static.source_height,
            patch_size=patch_size,
            cache_size=static.cache_size,
            max_motion=max_motion,
            random_enabled=random_enabled,
            rng=rng,
        )
        local_x, local_y = x - static.cache_x, y - static.cache_y
        cached = masks[
            static.cache_index,
            :,
            local_y : local_y + patch_size,
            local_x : local_x + patch_size,
        ]
        if segmentation_mode == "multiclass":
            foreground = (cached[by_name["loci"]] > threshold) | (
                cached[by_name["inoculum"]] > threshold
            )
            if merge_join_masks and "join" in by_name:
                foreground |= cached[by_name["join"]] > threshold
        else:
            foreground = cached[by_name["binary"]] > threshold
        foreground_pixels = int(foreground.sum())
        candidates_by_source.setdefault(static.source_id, []).append(
            CachedTrainingCropRecord(
                cache_index=static.cache_index,
                source_id=static.source_id,
                x=x,
                y=y,
                local_x=local_x,
                local_y=local_y,
                patch_size=patch_size,
                foreground_pixels=foreground_pixels,
                is_background_only=foreground_pixels == 0,
                soft_cldice_iterations=static.soft_cldice_iterations,
                scaled_width=static.source_width,
                scaled_height=static.source_height,
                source_crop_size=patch_size,
            )
        )

    retained: list[CachedTrainingCropRecord] = []
    for source_id, candidates in candidates_by_source.items():
        if not filter_empty:
            retained.extend(candidates)
            continue
        foreground_records = [
            record for record in candidates if record.foreground_pixels >= minimum
        ]
        background_pool = [
            record for record in candidates if record.is_background_only
        ]
        quota = (
            min(
                len(background_pool),
                max(
                    1,
                    int(
                        math.ceil(
                            len(foreground_records)
                            * background_percentage
                            / 100.0
                        )
                    ),
                ),
            )
            if background_enabled and background_pool
            else 0
        )
        selected_background: list[CachedTrainingCropRecord] = []
        if quota:
            rng = _seeded_rng(
                base_seed, fold_index, epoch, source_id, "background"
            )
            indices = rng.choice(
                len(background_pool), size=quota, replace=False
            )
            selected_background = [
                background_pool[int(index)] for index in indices
            ]
        retained_ids = {
            id(record)
            for record in foreground_records + selected_background
        }
        retained.extend(
            record for record in candidates if id(record) in retained_ids
        )
    return retained


class CachedSegmentationPatchDataset(Dataset):
    """Lazily crop final model inputs from run-level static memmaps."""

    def __init__(
        self,
        records: list[CachedTrainingCropRecord],
        cache: StaticPatchCache,
        transforms: A.Compose | None,
        segmentation_mode: str,
        mask_threshold: int,
        merge_join_masks: bool,
        target_weight_builder: Callable[[torch.Tensor], torch.Tensor] | None,
        source_soft_cldice_iterations: dict[str, int] | None,
        default_soft_cldice_iterations: int,
    ) -> None:
        self.records = records
        self.cache_dir = cache.cache_dir
        self.mask_names = cache.mask_names
        self.transforms = transforms
        self.segmentation_mode = segmentation_mode
        self.mask_threshold = int(mask_threshold)
        self.merge_join_masks = bool(merge_join_masks)
        self.target_weight_builder = target_weight_builder
        self.source_soft_cldice_iterations = source_soft_cldice_iterations
        self.default_soft_cldice_iterations = int(
            default_soft_cldice_iterations
        )
        self._images: np.ndarray | None = None
        self._masks: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._images is None:
            self._images = np.load(
                self.cache_dir / "images.npy", mmap_mode="r"
            )
            self._masks = np.load(
                self.cache_dir / "masks.npy", mmap_mode="r"
            )
        assert self._masks is not None
        return self._images, self._masks

    def __getitem__(self, index: int) -> dict[str, Any]:
        images, masks = self._arrays()
        record = self.records[index]
        y0, x0, size = record.local_y, record.local_x, record.patch_size
        image = np.array(
            images[
                record.cache_index,
                y0 : y0 + size,
                x0 : x0 + size,
            ],
            copy=True,
        )
        mask_stack = np.array(
            masks[
                record.cache_index,
                :,
                y0 : y0 + size,
                x0 : x0 + size,
            ],
            copy=True,
        )
        by_name = {
            name: mask_stack[channel]
            for channel, name in enumerate(self.mask_names)
        }
        if self.segmentation_mode == "multiclass":
            target, overlap = compose_multiclass_mask(
                by_name["loci"],
                by_name["inoculum"],
                self.mask_threshold,
                join_mask=by_name.get("join"),
                merge_join_masks=self.merge_join_masks,
            )
        else:
            target = (
                by_name["binary"] > self.mask_threshold
            ).astype(np.float32)
            overlap = {
                "overlap_pixels": 0,
                "overlap_fraction": 0.0,
            }

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=target)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
        else:
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            )
            mask_tensor = torch.from_numpy(target)

        if self.segmentation_mode == "multiclass":
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.squeeze(0)
            mask_tensor = mask_tensor.long()
        else:
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor[:1].float()

        sample: dict[str, Any] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "source_id": record.source_id,
            "x": record.x,
            "y": record.y,
            "patch_size": record.patch_size,
            "scale": record.scale,
            "scaled_width": record.scaled_width,
            "scaled_height": record.scaled_height,
            "resolution_bucket": record.resolution_bucket,
            "scale_label": record.scale_label,
            "foreground_pixels": record.foreground_pixels,
            "is_background_only": record.is_background_only,
            "source_crop_size": (
                record.source_crop_size or record.patch_size
            ),
            **overlap,
        }
        if self.target_weight_builder is not None:
            sample["loss_weight"] = self.target_weight_builder(
                mask_tensor
            ).float()
        if self.source_soft_cldice_iterations is not None:
            sample["soft_cldice_iterations"] = int(
                self.source_soft_cldice_iterations.get(
                    record.source_id,
                    self.default_soft_cldice_iterations,
                )
            )
        elif record.soft_cldice_iterations is not None:
            sample["soft_cldice_iterations"] = int(
                record.soft_cldice_iterations
            )
        return sample

    def cleanup(self) -> None:
        self._images = None
        self._masks = None
