from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "fungi_segmentation"},
    "segmentation": {
        "target": "loci",
    },
    "paths": {
        "images_dir": "data/images",
        "masks_dir": "data/masks",
        "mask_dirs": {
            "loci": "data/loci_masks",
            "inoculum": "data/inoculum_masks",
        },
        "runs_dir": "runs",
        "outputs_dir": "outputs",
    },
    "data": {
        "image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "pin_memory": True,
        "batch_size": 8,
        "image_size": None,
    },
    "patching": {
        "patch_size": 512,
        "overlap": 128,
        "stride": 384,
        "filter_empty_patches": True,
        "mask_threshold": 127,
        "min_foreground_pixels": 1,
        "image_resampling": "lanczos",
        "mask_resampling": "foreground_preserving",
        "train": {
            "random_offset": {
                "enabled": True,
                "max_fraction_of_patch": 0.5,
            },
            "scaled_context": {
                "enabled": True,
                "probability": 0.25,
                "max_scale": 2.0,
                "beta_alpha": 1.0,
                "beta_beta": 4.0,
            },
        },
        "validation": {
            "random_offset": {"enabled": False},
            "scaled_context": {"enabled": False},
        },
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
            "random_brightness_contrast": {
                "brightness_limit": [-0.2, 0.2],
                "contrast_limit": [-0.2, 0.2],
                "p": 0.3,
            },
            "random_gamma": {
                "gamma_limit": [90, 110],
                "p": 0.2,
            },
            "clahe": {
                "clip_limit": [1.0, 3.0],
                "tile_grid_size": [8, 8],
                "p": 0.15,
            },
            "blur": {
                "gaussian_blur_limit": [3, 5],
                "gaussian_sigma_limit": [0.1, 1.0],
                "defocus_radius": [1, 3],
                "defocus_alias_blur": [0.1, 0.3],
                "p": 0.2,
            },
            "gauss_noise_p": 0.1,
        },
    },
    "cv": {"n_splits": 5, "shuffle_groups": True, "random_state": 42},
    "split": {
        "mode": "csv",
        "csv_path": "data/image_splits.csv",
        "val_source_ids": [],
    },
    "model": {
        "name": "unetplusplus_resnet18",
        "in_channels": 3,
        "num_classes": 1,
        "encoder_name": "resnet18",
        "encoder_weights": "imagenet",
        "decoder_normalization": "instancenorm",
        "decoder_channels": [512, 256, 128, 64, 32],
        "decoder_attention_type": None,
    },
    "loss": {
        "name": "bce_dice_cldice",
        "bce_weight": 0.3,
        "dice_weight": 0.6,
        "soft_cldice_weight": 0.1,
        "iterations": 5,
        "smooth": 1e-6,
        "cldice_smooth": 1.0,
    },
    "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
    "scheduler": {
        "name": "reduce_on_plateau",
        "mode": "max",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1.0e-6,
        "monitor": "val_dice_macro_resolution",
    },
    "train": {
        "epochs": 50,
        "mixed_precision": True,
        "grad_clip": None,
        "monitor": "val_dice_macro_resolution",
        "monitor_mode": "max",
        "best_interval_checkpoint": {
            "enabled": True,
            "interval_epochs": 10,
        },
        "threshold": 0.5,
        "enable_per_image_validation": True,
        "per_image_validation_interval": 1,
        "seed": 42,
        "device": "auto",
        "use_tqdm": True,
    },
    "inference": {"threshold": 0.5, "save_probabilities": False},
    "test_evaluation": {
        "enabled": True,
        "threshold_start": 0.5,
        "threshold_stop": 1.0,
        "threshold_step": 0.01,
        "cldice_iterations": 3,
    },
    "qualitative_evaluation": {
        "enabled": True,
        "data_root": None,
        "split": "test",
        "crop_patch_grid": [3, 3],
        "min_foreground_ratio": 0.005,
        "max_foreground_ratio": 0.15,
        "selection_seed": 42,
        "max_checkpoints": None,
    },
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
    loaded_paths = loaded.get("paths", {}) if isinstance(loaded.get("paths"), dict) else {}
    if (
        "segmentation" not in loaded
        and "masks_dir" in loaded_paths
        and "mask_dirs" not in loaded_paths
    ):
        config["segmentation"]["target"] = "legacy"

    if "patching" not in loaded and isinstance(loaded.get("data"), dict):
        legacy_data = loaded["data"]
        for key in (
            "patch_size",
            "overlap",
            "stride",
            "filter_empty_patches",
            "mask_threshold",
            "min_foreground_pixels",
        ):
            if key in legacy_data:
                config["patching"][key] = legacy_data[key]
        multiscale = legacy_data.get("multiscale", {})
        if isinstance(multiscale, dict):
            for key in ("image_resampling", "mask_resampling"):
                if key in multiscale:
                    config["patching"][key] = multiscale[key]

    patching_cfg = config["patching"]
    patching_cfg["stride"] = int(
        patching_cfg.get("stride")
        or (int(patching_cfg["patch_size"]) - int(patching_cfg["overlap"]))
    )
    return config


def resolve_mask_dir(config: dict[str, Any]) -> Path:
    target = str(config.get("segmentation", {}).get("target", "")).strip()
    mask_dirs = config.get("paths", {}).get("mask_dirs", {})
    if target and isinstance(mask_dirs, dict) and target in mask_dirs:
        return Path(mask_dirs[target])

    legacy_masks_dir = config.get("paths", {}).get("masks_dir")
    if legacy_masks_dir:
        return Path(legacy_masks_dir)

    available_targets = ", ".join(sorted(str(key) for key in mask_dirs)) if isinstance(mask_dirs, dict) else ""
    raise ValueError(
        f"No mask directory configured for segmentation target '{target}'. "
        f"Available targets: {available_targets or 'none'}."
    )
