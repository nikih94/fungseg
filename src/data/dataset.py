from __future__ import annotations

from typing import Any, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.patching import PatchRecord, crop_and_pad_array


def _build_normalize(augmentations_config: Optional[dict[str, Any]]) -> A.Normalize:
    normalize_config = (augmentations_config or {}).get("normalize", {})
    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])
    return A.Normalize(mean=tuple(mean), std=tuple(std))


def get_train_transforms(
    image_size: Optional[int] = None,
    augmentations_config: Optional[dict[str, Any]] = None,
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
    return A.Compose(ops)


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
    ) -> None:
        self.records = records
        self.mask_threshold = mask_threshold
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        with Image.open(record.image_path) as image:
            image_array = np.array(image.convert("RGB"))

        with Image.open(record.mask_path) as mask:
            mask_array = np.array(mask.convert("L"), dtype=np.uint8)

        image_patch = crop_and_pad_array(image_array, record.x, record.y, record.patch_size)
        mask_patch = crop_and_pad_array(mask_array, record.x, record.y, record.patch_size)

        binary_mask = (mask_patch > self.mask_threshold).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image_patch, mask=binary_mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
        else:
            image_tensor = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask)

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        else:
            mask_tensor = mask_tensor[:1]
        mask_tensor = mask_tensor.float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "source_id": record.source_id,
            "x": record.x,
            "y": record.y,
            "patch_size": record.patch_size,
        }
