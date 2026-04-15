from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "fungi_segmentation"},
    "paths": {
        "images_dir": "data/images",
        "masks_dir": "data/masks",
        "runs_dir": "runs",
        "outputs_dir": "outputs",
    },
    "data": {
        "image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "patch_size": 512,
        "overlap": 128,
        "stride": 384,
        "filter_empty_patches": True,
        "mask_threshold": 127,
        "min_foreground_pixels": 1,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
        "batch_size": 8,
        "image_size": None,
    },
    "augmentations": {
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "train": {
            "horizontal_flip_p": 0.8,
            "vertical_flip_p": 0.6,
            "random_rotate_90_p": 0.8,
            "affine": {
                "translate_x": [-0.05, 0.05],
                "translate_y": [-0.05, 0.05],
                "scale": [0.9, 1.1],
                "rotate": [-30, 30],
                "p": 0.6,
            },
            "random_brightness_contrast_p": 0.3,
            "gauss_noise_p": 0.2,
        },
    },
    "cv": {"n_splits": 5, "shuffle_groups": True, "random_state": 42},
    "split": {
        "mode": "train_val",
        "val_source_ids": [],
    },
    "model": {
        "name": "unetplusplus_resnet18",
        "in_channels": 3,
        "num_classes": 1,
        "encoder_name": "resnet18",
        "encoder_weights": "imagenet",
        "decoder_channels": [512, 256, 128, 64, 32],
    },
    "loss": {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5},
    "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
    "scheduler": {
        "name": "reduce_on_plateau",
        "mode": "max",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1.0e-6,
        "monitor": "val_dice_per_patch",
    },
    "train": {
        "epochs": 50,
        "mixed_precision": True,
        "grad_clip": None,
        "monitor": "val_dice_per_patch",
        "monitor_mode": "max",
        "threshold": 0.5,
        "enable_per_image_validation": True,
        "per_image_validation_interval": 1,
        "seed": 42,
        "device": "auto",
        "use_tqdm": True,
    },
    "inference": {"threshold": 0.5, "save_probabilities": False},
}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = _deep_update(deepcopy(DEFAULT_CONFIG), loaded)
    data_cfg = config["data"]
    data_cfg["stride"] = int(data_cfg.get("stride") or (int(data_cfg["patch_size"]) - int(data_cfg["overlap"])))
    return config
