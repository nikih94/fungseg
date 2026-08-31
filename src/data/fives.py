from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import albumentations as A
from PIL import Image
import torch

from src.data.dataset import SegmentationPatchDataset
from src.patching import PatchRecord


FIVES_ROOT = Path("data/FIVES")
FIVES_IMAGES_DIR = FIVES_ROOT / "Original"
FIVES_MASKS_DIR = FIVES_ROOT / "Ground truth"


def discover_fives_pairs(
    image_extensions: Iterable[str],
    *,
    images_dir: Path = FIVES_IMAGES_DIR,
    masks_dir: Path = FIVES_MASKS_DIR,
) -> list[tuple[Path, Path]]:
    """Return complete FIVES image/mask pairs, failing on an invalid enabled dataset."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"FIVES image directory does not exist: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"FIVES mask directory does not exist: {masks_dir}")

    extensions = {str(extension).lower() for extension in image_extensions}
    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }
    masks = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for: {', '.join(missing_masks)}")
        if missing_images:
            details.append(f"masks without images for: {', '.join(missing_images)}")
        raise ValueError("Incomplete FIVES image/mask pairs; " + "; ".join(details))
    if not images:
        raise RuntimeError(f"No FIVES image/mask pairs were found under {images_dir.parent}")
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def centered_fives_coordinates(width: int, height: int, patch_size: int) -> list[tuple[int, int]]:
    """Return a centered, non-overlapping 2x2 patch grid."""
    if patch_size <= 0:
        raise ValueError("FIVES patch size must be positive.")
    required_extent = 2 * patch_size
    if width < required_extent or height < required_extent:
        raise ValueError(
            "FIVES images must fit a centered 2x2 patch grid: "
            f"image={width}x{height}, patch_size={patch_size}."
        )
    left = (width - required_extent) // 2
    top = (height - required_extent) // 2
    return [
        (left, top),
        (left + patch_size, top),
        (left, top + patch_size),
        (left + patch_size, top + patch_size),
    ]


def build_fives_patch_records(
    pairs: Iterable[tuple[Path, Path]],
    patch_size: int,
) -> list[PatchRecord]:
    records: list[PatchRecord] = []
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        if mask_size != (width, height):
            raise ValueError(
                f"FIVES image/mask dimensions differ for {image_path.name}: "
                f"image={width}x{height}, mask={mask_size[0]}x{mask_size[1]}."
            )
        for x, y in centered_fives_coordinates(width, height, patch_size):
            records.append(
                PatchRecord(
                    source_id=f"FIVES/{image_path.name}",
                    image_path=image_path,
                    mask_path=mask_path,
                    x=x,
                    y=y,
                    patch_size=patch_size,
                    scale=1.0,
                    scaled_width=width,
                    scaled_height=height,
                    resolution_bucket="fives_center",
                    scale_label="fives_center",
                    source_crop_size=patch_size,
                )
            )
    return records


def load_fives_training_records(config: dict[str, Any]) -> list[PatchRecord]:
    if not bool(config.get("data", {}).get("use_fives", False)):
        return []
    pairs = discover_fives_pairs(config["data"]["image_extensions"])
    return build_fives_patch_records(pairs, int(config["patching"]["patch_size"]))


class FivesPatchDataset(SegmentationPatchDataset):
    """Binary vessel patches, optionally exposed as multiclass loci targets."""

    def __init__(
        self,
        records: list[PatchRecord],
        mask_threshold: int,
        transforms: Optional[A.Compose] = None,
        *,
        segmentation_mode: str = "binary",
        target_weight_builder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        soft_cldice_iterations: dict[str, int] | None = None,
        default_soft_cldice_iterations: int = 0,
    ) -> None:
        super().__init__(
            records=records,
            mask_threshold=mask_threshold,
            transforms=transforms,
            segmentation_mode="binary",
            target_weight_builder=target_weight_builder,
            soft_cldice_iterations=soft_cldice_iterations,
            default_soft_cldice_iterations=default_soft_cldice_iterations,
        )
        self.output_segmentation_mode = str(segmentation_mode).lower()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        if self.output_segmentation_mode == "multiclass":
            sample["mask"] = sample["mask"].squeeze(0).long()
        return sample
