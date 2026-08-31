from __future__ import annotations

from typing import Any, Callable, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.patching import PatchRecord, crop_scaled_image_patch, crop_scaled_mask_patch


def compose_multiclass_mask(
    loci_mask: np.ndarray,
    inoculum_mask: np.ndarray,
    threshold: int = 127,
    join_mask: np.ndarray | None = None,
    merge_join_masks: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    loci = loci_mask > threshold
    inoculum = inoculum_mask > threshold
    join = None if join_mask is None else join_mask > threshold
    if merge_join_masks and join is not None:
        loci = loci | join
    overlap_pixels = int((loci & inoculum).sum())
    mask = np.zeros(loci.shape, dtype=np.uint8)
    mask[loci] = 1
    mask[inoculum] = 2
    return mask, {
        "overlap_pixels": overlap_pixels,
        "overlap_fraction": overlap_pixels / max(int(mask.size), 1),
    }


def _build_normalize(augmentations_config: Optional[dict[str, Any]]) -> A.Normalize:
    normalize_config = (augmentations_config or {}).get("normalize", {})
    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])
    return A.Normalize(mean=tuple(mean), std=tuple(std))


def get_train_transforms(
    image_size: Optional[int] = None,
    augmentations_config: Optional[dict[str, Any]] = None,
    seed: int | None = None,
) -> A.Compose:
    train_config = (augmentations_config or {}).get("train", {})
    affine_config = train_config.get("affine", {})
    brightness_contrast_config = train_config.get("random_brightness_contrast", {})
    gamma_config = train_config.get("random_gamma", {})
    clahe_config = train_config.get("clahe", {})
    blur_config = train_config.get("blur", {})
    ops = []
    if image_size is not None:
        ops.append(A.Resize(image_size, image_size))
    ops.extend(
        [
            A.HorizontalFlip(p=float(train_config.get("horizontal_flip_p", 0.8))),
            A.VerticalFlip(p=float(train_config.get("vertical_flip_p", 0.6))),
            A.RandomRotate90(p=float(train_config.get("random_rotate_90_p", 0.8))),
            A.Affine(
                translate_percent={
                    "x": tuple(affine_config.get("translate_x", [-0.05, 0.05])),
                    "y": tuple(affine_config.get("translate_y", [-0.05, 0.05])),
                },
                scale=tuple(affine_config.get("scale", [0.9, 1.1])),
                rotate=tuple(affine_config.get("rotate", [-30, 30])),
                p=float(affine_config.get("p", 0.6)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=tuple(brightness_contrast_config.get("brightness_limit", [-0.2, 0.2])),
                contrast_limit=tuple(brightness_contrast_config.get("contrast_limit", [-0.2, 0.2])),
                p=float(brightness_contrast_config.get("p", train_config.get("random_brightness_contrast_p", 0.3))),
            ),
            A.RandomGamma(
                gamma_limit=tuple(gamma_config.get("gamma_limit", [90, 110])),
                p=float(gamma_config.get("p", 0.2)),
            ),
            A.CLAHE(
                clip_limit=tuple(clahe_config.get("clip_limit", [1.0, 3.0])),
                tile_grid_size=tuple(clahe_config.get("tile_grid_size", [8, 8])),
                p=float(clahe_config.get("p", 0.15)),
            ),
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=tuple(blur_config.get("gaussian_blur_limit", [3, 5])),
                        sigma_limit=tuple(blur_config.get("gaussian_sigma_limit", [0.1, 1.0])),
                        p=1.0,
                    ),
                    A.Defocus(
                        radius=tuple(blur_config.get("defocus_radius", [1, 3])),
                        alias_blur=tuple(blur_config.get("defocus_alias_blur", [0.1, 0.3])),
                        p=1.0,
                    ),
                ],
                p=float(blur_config.get("p", 0.2)),
            ),
            A.GaussNoise(p=float(train_config.get("gauss_noise_p", 0.2))),
            _build_normalize(augmentations_config),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops, seed=seed)


def get_val_transforms(
    image_size: Optional[int] = None,
    augmentations_config: Optional[dict[str, Any]] = None,
) -> A.Compose:
    ops = []
    if image_size is not None:
        ops.append(A.Resize(image_size, image_size))
    ops.extend(
        [
            _build_normalize(augmentations_config),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops)


class SegmentationPatchDataset(Dataset):
    def __init__(
        self,
        records: list[PatchRecord],
        mask_threshold: int,
        transforms: Optional[A.Compose] = None,
        image_resampling: str = "lanczos",
        mask_resampling: str = "foreground_preserving",
        segmentation_mode: str = "binary",
        target_weight_builder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        merge_join_masks: bool = False,
        soft_cldice_iterations: dict[str, int] | None = None,
        default_soft_cldice_iterations: int = 0,
    ) -> None:
        self.records = records
        self.mask_threshold = mask_threshold
        self.transforms = transforms
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.segmentation_mode = str(segmentation_mode).lower()
        self.target_weight_builder = target_weight_builder
        self.merge_join_masks = bool(merge_join_masks)
        self.soft_cldice_iterations = soft_cldice_iterations
        self.default_soft_cldice_iterations = int(default_soft_cldice_iterations)

    def __len__(self) -> int:
        return len(self.records)

    def set_records(self, records: list[PatchRecord]) -> None:
        self.records = records

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        with Image.open(record.image_path) as image:
            image_array = np.array(image.convert("RGB"))

        mask_arrays: dict[str, np.ndarray] = {}
        if self.segmentation_mode == "multiclass":
            if not record.mask_paths or "loci" not in record.mask_paths or "inoculum" not in record.mask_paths:
                raise ValueError("Multiclass records require named loci and inoculum mask paths.")
            for name in ("loci", "inoculum", "join"):
                mask_path = record.mask_paths.get(name)
                if mask_path is not None:
                    with Image.open(mask_path) as mask:
                        mask_arrays[name] = np.array(mask.convert("L"), dtype=np.uint8)
        else:
            with Image.open(record.mask_path) as mask:
                mask_arrays["binary"] = np.array(mask.convert("L"), dtype=np.uint8)

        image_patch = crop_scaled_image_patch(
            image_array,
            x=record.x,
            y=record.y,
            patch_size=record.patch_size,
            scale=record.scale,
            resampling=self.image_resampling,
        )
        mask_patches = {
            name: crop_scaled_mask_patch(
                array, x=record.x, y=record.y, patch_size=record.patch_size,
                scale=record.scale, mask_threshold=self.mask_threshold,
                resampling=self.mask_resampling,
            ) for name, array in mask_arrays.items()
        }
        if self.segmentation_mode == "multiclass":
            target_mask, overlap = compose_multiclass_mask(
                mask_patches["loci"],
                mask_patches["inoculum"],
                self.mask_threshold,
                join_mask=mask_patches.get("join"),
                merge_join_masks=self.merge_join_masks,
            )
        else:
            target_mask = (mask_patches["binary"] > self.mask_threshold).astype(np.float32)
            overlap = {"overlap_pixels": 0, "overlap_fraction": 0.0}

        if self.transforms is not None:
            transformed = self.transforms(image=image_patch, mask=target_mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
        else:
            image_tensor = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(target_mask)

        if self.segmentation_mode == "multiclass":
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.squeeze(0)
            mask_tensor = mask_tensor.long()
        else:
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            else:
                mask_tensor = mask_tensor[:1]
            mask_tensor = mask_tensor.float()

        sample = {
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
            "source_crop_size": record.source_crop_size or record.patch_size,
            **overlap,
        }
        if self.target_weight_builder is not None:
            sample["loss_weight"] = self.target_weight_builder(mask_tensor).float()
        if self.soft_cldice_iterations is not None:
            sample["soft_cldice_iterations"] = int(
                self.soft_cldice_iterations.get(
                    record.source_id, self.default_soft_cldice_iterations
                )
            )
        return sample
