from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "fungi_segmentation"},
    "segmentation": {
        "mode": "binary",
        "target": "loci",
        "classes": {"background": 0, "loci": 1, "inoculum": 2},
        "overlap_precedence": "inoculum",
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
    "join_masks": {
        "enabled": False,
        "masks_dir": "data/join_masks",
        "merge_with_loci": False,
        "evaluation_enabled": False,
    },
    "data": {
        "image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "use_fives": False,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "pin_memory": True,
        "batch_size": 8,
        "image_size": None,
        "train_patch_cache": {"enabled": True},
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
            "background_only": {
                "enabled": False,
                "percentage_of_foreground": 5.0,
            },
            "scaled_context": {
                "enabled": False,
                "probability": 0.25,
                "max_scale": 2.0,
                "beta_alpha": 1.0,
                "beta_beta": 4.0,
                "containment_filter": {
                    "enabled": False,
                    "threshold": 0.8,
                    "preserve_normal_patches": True,
                },
            },
        },
        "validation": {
            "random_offset": {"enabled": False},
            "scaled_context": {"enabled": False},
        },
    },
    "validation": {
        "start_epoch": 1,
        "full_image": {
            "enabled": True,
            "batch_size": 1,
            "interval_epochs": 1,
            "selection": "all",
            "max_images": None,
            "soft_cldice_foreground_only": True,
            "patch_cache": {"enabled": False},
            "loss": {"patch_selection": "non_overlapping"},
            "composite_metrics": {
                "dice_cldice_per_image": {
                    "weights": {
                        "dice_per_image": 0.7,
                        "cldice_per_image": 0.3,
                    },
                },
            },
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
        "shallow_channels": [16, 32],
        "refine_half_channels": [128, 64],
        "refine_full_channels": [32, 32],
    },
    "loss": {
        "name": "bce_dice_cldice",
        "bce_weight": 0.3,
        "dice_weight": 0.6,
        "soft_cldice_weight": 0.1,
        "iterations": 5,
        "iterations_csv": None,
        "static_patch_iterations": {
            "enabled": True,
            "margin_iterations": 10,
            "round_up_to": 10,
        },
        "smooth": 1e-6,
        "cldice_smooth": 1.0,
    },
    "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
    "scheduler": {
        "name": "reduce_on_plateau",
        "mode": "max",
        "factor": 0.5,
        "patience": 7,
        "threshold": 0.001,
        "threshold_mode": "abs",
        "min_lr": 1.0e-6,
        "monitor": "val_dice_cldice_per_image",
    },
    "checkpointing": {
        "primary": "current",
        "save_last": True,
        "interval": {"enabled": True, "interval_epochs": 10},
        "selections": {
            "current": {
                "enabled": True,
                "filename": "best_current.pt",
                "monitor": "val_dice_cldice_per_image",
                "mode": "max",
            },
            "dice": {
                "enabled": True,
                "filename": "best_dice.pt",
                "monitor": "val_dice_per_image",
                "mode": "max",
            },
            "validation_loss": {
                "enabled": False,
                "filename": "best_val_loss.pt",
                "monitor": "val_loss",
                "mode": "min",
            },
        },
    },
    "train": {
        "epochs": 50,
        "mixed_precision": True,
        "grad_clip": None,
        "threshold": 0.5,
        "seed": 42,
        "device": "auto",
        "use_tqdm": True,
        "compute_hard_cldice_metrics": False,
    },
    "inference": {"threshold": 0.5, "save_probabilities": False},
    "test_evaluation": {
        "enabled": True,
        "threshold_start": 0.5,
        "threshold_stop": 1.0,
        "threshold_step": 0.01,
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


def _normalize_validation_config(
    config: dict[str, Any],
    loaded: dict[str, Any],
) -> None:
    train = config["train"]
    loaded_validation = loaded.get("validation")
    if loaded_validation is not None and not isinstance(loaded_validation, dict):
        raise ValueError("validation must be a mapping.")

    if loaded_validation is None:
        legacy_selection = train.get("full_image_validation", {})
        if not isinstance(legacy_selection, dict):
            raise ValueError("train.full_image_validation must be a mapping.")
        legacy_monitor = train.get("full_image_monitor", {})
        if not isinstance(legacy_monitor, dict):
            raise ValueError("train.full_image_monitor must be a mapping.")
        config["validation"] = {
            "start_epoch": 1,
            "full_image": {
                "enabled": bool(train.get("enable_per_image_validation", True)),
                "batch_size": 1,
                "interval_epochs": int(
                    train.get("per_image_validation_interval", 1)
                ),
                "selection": legacy_selection.get("selection", "all"),
                "max_images": legacy_selection.get("max_images"),
                "loss": {"patch_selection": "non_overlapping"},
                "composite_metrics": {
                    "dice_cldice_per_image": {
                        "weights": {
                            "dice_per_image": legacy_monitor.get("dice_weight", 0.7),
                            "cldice_per_image": legacy_monitor.get("cldice_weight", 0.3),
                        },
                    },
                },
            },
        }

    validation = config["validation"]
    full_image = validation.get("full_image", {})
    if not isinstance(full_image, dict):
        raise ValueError("validation.full_image must be a mapping.")

    start_epoch = int(validation.get("start_epoch", 1))
    if start_epoch <= 0:
        raise ValueError("validation.start_epoch must be positive.")
    if start_epoch > int(train["epochs"]):
        raise ValueError("validation.start_epoch must not exceed train.epochs.")
    validation["start_epoch"] = start_epoch

    interval_epochs = int(full_image.get("interval_epochs", 1))
    if interval_epochs <= 0:
        raise ValueError("validation.full_image.interval_epochs must be positive.")
    batch_size = int(full_image.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("validation.full_image.batch_size must be positive.")
    selection = str(full_image.get("selection", "all")).strip().lower()
    if selection not in {"all", "smallest_area"}:
        raise ValueError(
            "validation.full_image.selection must be 'all' or 'smallest_area'."
        )
    max_images = full_image.get("max_images")
    if max_images is not None and int(max_images) <= 0:
        raise ValueError("validation.full_image.max_images must be positive.")
    if selection == "smallest_area" and max_images is None:
        raise ValueError(
            "validation.full_image.max_images is required for smallest_area selection."
        )
    patch_cache = full_image.get("patch_cache", {})
    if not isinstance(patch_cache, dict):
        raise ValueError(
            "validation.full_image.patch_cache must be a mapping."
        )
    loss = full_image.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("validation.full_image.loss must be a mapping.")
    patch_selection = str(
        loss.get("patch_selection", "non_overlapping")
    ).strip().lower()
    if patch_selection != "non_overlapping":
        raise ValueError(
            "validation.full_image.loss.patch_selection must be "
            "'non_overlapping'."
        )

    loaded_full_image = (
        loaded_validation.get("full_image", {})
        if isinstance(loaded_validation, dict)
        else {}
    )
    if not isinstance(loaded_full_image, dict):
        loaded_full_image = {}
    legacy_monitor = loaded_full_image.get("monitor")
    if "composite_metrics" not in loaded_full_image and isinstance(legacy_monitor, dict):
        full_image["composite_metrics"] = {
            "dice_cldice_per_image": {
                "weights": {
                    "dice_per_image": legacy_monitor.get("dice_weight", 0.7),
                    "cldice_per_image": legacy_monitor.get("cldice_weight", 0.3),
                },
            },
        }
    composites = full_image.get("composite_metrics", {})
    if not isinstance(composites, dict) or not composites:
        raise ValueError(
            "validation.full_image.composite_metrics must be a non-empty mapping."
        )
    if str(config["segmentation"].get("mode", "binary")).lower() == "multiclass":
        composites.setdefault(
            "dice_low_cldice_per_image",
            {"weights": {"dice_per_image": 0.9, "cldice_per_image": 0.1}},
        )
        composites.setdefault(
            "inoculum_compensated_per_image",
            {
                "weights": {
                    "dice_loci_per_image": 0.3,
                    "dice_inoculum_per_image": 0.5,
                    "cldice_per_image": 0.2,
                },
            },
        )
    normalized_composites: dict[str, dict[str, dict[str, float]]] = {}
    for name, definition in composites.items():
        if not isinstance(definition, dict):
            raise ValueError(
                f"validation.full_image.composite_metrics.{name} must be a mapping."
            )
        weights = definition.get("weights", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                f"validation.full_image.composite_metrics.{name}.weights must "
                "be a non-empty mapping."
            )
        normalized_weights = {
            str(metric): float(weight) for metric, weight in weights.items()
        }
        if any(weight < 0.0 for weight in normalized_weights.values()):
            raise ValueError(f"Composite metric {name!r} has a negative weight.")
        if abs(sum(normalized_weights.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"Composite metric {name!r} weights must sum to 1.0."
            )
        normalized_composites[str(name)] = {"weights": normalized_weights}
    full_image["enabled"] = bool(full_image.get("enabled", True))
    full_image["batch_size"] = batch_size
    full_image["interval_epochs"] = interval_epochs
    full_image["selection"] = selection
    full_image["max_images"] = None if max_images is None else int(max_images)
    full_image["soft_cldice_foreground_only"] = bool(
        full_image.get("soft_cldice_foreground_only", True)
    )
    full_image["patch_cache"] = {
        "enabled": bool(patch_cache.get("enabled", False))
    }
    full_image["loss"] = {"patch_selection": patch_selection}
    full_image["composite_metrics"] = normalized_composites
    full_image.pop("monitor", None)
    validation.pop("fast", None)
    validation["full_image"] = full_image

    for legacy_key in (
        "enable_per_image_validation",
        "per_image_validation_interval",
        "full_image_validation",
        "full_image_monitor",
    ):
        train.pop(legacy_key, None)



def _normalize_checkpointing_config(
    config: dict[str, Any], loaded: dict[str, Any]
) -> None:
    train = config["train"]
    loaded_checkpointing = loaded.get("checkpointing")
    if loaded_checkpointing is not None and not isinstance(loaded_checkpointing, dict):
        raise ValueError("checkpointing must be a mapping.")

    if loaded_checkpointing is None:
        selections: dict[str, dict[str, Any]] = {
            "current": {
                "enabled": True,
                "filename": "best_current.pt",
                "monitor": train.get("monitor", "val_dice_cldice_per_image"),
                "mode": train.get("monitor_mode", "max"),
            },
            "dice": {
                "enabled": True,
                "filename": "best_dice.pt",
                "monitor": "val_dice_per_image",
                "mode": "max",
            },
            "validation_loss": {
                "enabled": True,
                "filename": "best_val_loss.pt",
                "monitor": "val_loss",
                "mode": "min",
            },
        }
        if str(config["segmentation"].get("mode", "binary")).lower() == "multiclass":
            selections.update({
                "low_cldice": {
                    "enabled": True,
                    "filename": "best_low_cldice.pt",
                    "monitor": "val_dice_low_cldice_per_image",
                    "mode": "max",
                },
                "inoculum_compensated": {
                    "enabled": True,
                    "filename": "best_inoculum_compensated.pt",
                    "monitor": "val_inoculum_compensated_per_image",
                    "mode": "max",
                },
            })
        legacy_interval = train.get("best_interval_checkpoint", {})
        if not isinstance(legacy_interval, dict):
            raise ValueError("train.best_interval_checkpoint must be a mapping.")
        checkpointing = {
            "primary": "current",
            "save_last": bool(train.get("save_last_checkpoint", True)),
            "interval": {
                "enabled": bool(legacy_interval.get("enabled", True)),
                "interval_epochs": int(legacy_interval.get("interval_epochs", 10)),
            },
            "selections": selections,
        }
    else:
        checkpointing = deepcopy(loaded_checkpointing)
        checkpointing.setdefault("primary", "current")
        checkpointing.setdefault("save_last", True)
        checkpointing.setdefault(
            "interval", {"enabled": True, "interval_epochs": 10}
        )

    interval = checkpointing.get("interval", {})
    if not isinstance(interval, dict):
        raise ValueError("checkpointing.interval must be a mapping.")
    interval_epochs = int(interval.get("interval_epochs", 10))
    if interval_epochs <= 0:
        raise ValueError("checkpointing.interval.interval_epochs must be positive.")
    checkpointing["interval"] = {
        "enabled": bool(interval.get("enabled", True)),
        "interval_epochs": interval_epochs,
    }

    selections = checkpointing.get("selections", {})
    if not isinstance(selections, dict) or not selections:
        raise ValueError("checkpointing.selections must be a non-empty mapping.")
    normalized_selections: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for name, definition in selections.items():
        if not isinstance(definition, dict):
            raise ValueError(f"checkpointing.selections.{name} must be a mapping.")
        filename = str(definition.get("filename", "")).strip()
        if not filename or Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError(
                f"checkpointing.selections.{name}.filename must be a .pt basename."
            )
        if filename in filenames:
            raise ValueError(f"Duplicate checkpoint filename: {filename}")
        filenames.add(filename)
        mode = str(definition.get("mode", "max")).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError(
                f"checkpointing.selections.{name}.mode must be 'min' or 'max'."
            )
        monitor = str(definition.get("monitor", "")).strip()
        if not monitor:
            raise ValueError(f"checkpointing.selections.{name}.monitor is required.")
        normalized_selections[str(name)] = {
            "enabled": bool(definition.get("enabled", True)),
            "filename": filename,
            "monitor": monitor,
            "mode": mode,
        }
    primary = str(checkpointing.get("primary", "current"))
    if primary not in normalized_selections:
        raise ValueError("checkpointing.primary must name a configured selection.")
    if not normalized_selections[primary]["enabled"]:
        raise ValueError("checkpointing.primary selection must be enabled.")
    checkpointing["primary"] = primary
    checkpointing["save_last"] = bool(checkpointing.get("save_last", True))
    checkpointing["selections"] = normalized_selections
    config["checkpointing"] = checkpointing

    for legacy_key in (
        "monitor", "monitor_mode", "best_interval_checkpoint", "save_last_checkpoint"
    ):
        train.pop(legacy_key, None)


def _normalize_join_masks_config(config: dict[str, Any]) -> None:
    join_masks = config.get("join_masks", {})
    if not isinstance(join_masks, dict):
        raise ValueError("join_masks must be a mapping.")
    enabled = bool(join_masks.get("enabled", False))
    merge_with_loci = bool(join_masks.get("merge_with_loci", False))
    evaluation_enabled = bool(join_masks.get("evaluation_enabled", False))
    masks_dir = str(join_masks.get("masks_dir", "")).strip()
    if (enabled or evaluation_enabled) and not masks_dir:
        raise ValueError(
            "join_masks.masks_dir is required when join masks are enabled for "
            "training or evaluation."
        )
    if merge_with_loci and not enabled:
        raise ValueError("join_masks.merge_with_loci requires join_masks.enabled: true.")
    config["join_masks"] = {
        "enabled": enabled,
        "masks_dir": masks_dir,
        "merge_with_loci": merge_with_loci,
        "evaluation_enabled": evaluation_enabled,
    }


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = _deep_update(deepcopy(DEFAULT_CONFIG), loaded)
    loaded_optimizer = loaded.get("optimizer", {})
    if isinstance(loaded_optimizer, dict):
        has_encoder_lr = "encoder_lr" in loaded_optimizer
        has_decoder_lr = "decoder_lr" in loaded_optimizer
        if has_encoder_lr != has_decoder_lr:
            raise ValueError(
                "optimizer.encoder_lr and optimizer.decoder_lr must be configured together."
            )
        if has_encoder_lr:
            config["optimizer"].pop("lr", None)
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
    patch_size = int(patching_cfg["patch_size"])
    overlap = int(patching_cfg["overlap"])
    stride = int(patching_cfg["stride"])
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patching.patch_size and patching.stride must be positive.")
    if overlap < 0:
        raise ValueError("patching.overlap must be non-negative.")
    cache_cfg = config["data"].get("train_patch_cache", {})
    if not isinstance(cache_cfg, dict):
        raise ValueError("data.train_patch_cache must be a mapping.")
    cache_cfg["enabled"] = bool(cache_cfg.get("enabled", True))
    config["data"]["train_patch_cache"] = cache_cfg
    train_patching_cfg = patching_cfg.get("train", {})
    random_offset_cfg = train_patching_cfg.get("random_offset", {})
    scaled_context_cfg = train_patching_cfg.get("scaled_context", {})
    if cache_cfg["enabled"] and bool(scaled_context_cfg.get("enabled", False)):
        raise ValueError(
            "Static training patch caching is incompatible with "
            "patching.train.scaled_context.enabled: true."
        )
    if (
        cache_cfg["enabled"]
        and bool(random_offset_cfg.get("enabled", False))
        and overlap <= 0
    ):
        raise ValueError(
            "Cached training random offsets require patching.overlap to be positive."
        )
    static_iteration_cfg = config["loss"].get("static_patch_iterations", {})
    if not isinstance(static_iteration_cfg, dict):
        raise ValueError("loss.static_patch_iterations must be a mapping.")
    margin_iterations = int(static_iteration_cfg.get("margin_iterations", 10))
    round_up_to = int(static_iteration_cfg.get("round_up_to", 10))
    if margin_iterations < 0:
        raise ValueError(
            "loss.static_patch_iterations.margin_iterations must be non-negative."
        )
    if round_up_to <= 0:
        raise ValueError("loss.static_patch_iterations.round_up_to must be positive.")
    config["loss"]["static_patch_iterations"] = {
        "enabled": bool(static_iteration_cfg.get("enabled", True)),
        "margin_iterations": margin_iterations,
        "round_up_to": round_up_to,
    }
    background_only_cfg = patching_cfg.get("train", {}).get(
        "background_only", {}
    )
    if not isinstance(background_only_cfg, dict):
        raise ValueError("patching.train.background_only must be a mapping.")
    background_percentage = float(
        background_only_cfg.get("percentage_of_foreground", 0.0)
    )
    if not 0.0 <= background_percentage <= 100.0:
        raise ValueError(
            "patching.train.background_only.percentage_of_foreground must be "
            "between 0 and 100."
        )
    background_only_cfg["enabled"] = bool(
        background_only_cfg.get("enabled", False)
    )
    background_only_cfg["percentage_of_foreground"] = background_percentage
    _normalize_validation_config(config, loaded)
    _normalize_checkpointing_config(config, loaded)
    _normalize_join_masks_config(config)
    return config


def _retain_known_relevant_keys(
    section: dict[str, Any],
    known_conditional_keys: set[str],
    relevant_keys: set[str],
) -> None:
    for key in known_conditional_keys - relevant_keys:
        section.pop(key, None)


def config_for_persistence(
    config: dict[str, Any],
    *,
    training_date: str | None = None,
) -> dict[str, Any]:
    """Return an effective config without defaults inactive for the selected pipeline."""
    persisted = deepcopy(config)
    segmentation = persisted.get("segmentation", {})
    segmentation_mode = str(segmentation.get("mode", "binary")).lower()
    paths = persisted.get("paths", {})
    join_masks = persisted.get("join_masks", {})
    if not bool(join_masks.get("evaluation_enabled", False)):
        join_masks.pop("evaluation_enabled", None)
    if segmentation_mode != "multiclass" or not (
        bool(join_masks.get("enabled", False))
        or bool(join_masks.get("evaluation_enabled", False))
    ):
        persisted.pop("join_masks", None)

    if segmentation_mode == "multiclass":
        segmentation.pop("target", None)
        paths.pop("masks_dir", None)
        persisted.get("train", {}).pop("threshold", None)
        persisted.get("inference", {}).pop("threshold", None)
        for key in ("threshold_start", "threshold_stop", "threshold_step", "threshold_sweep"):
            persisted.get("test_evaluation", {}).pop(key, None)
    else:
        segmentation.pop("classes", None)
        segmentation.pop("overlap_precedence", None)
        persisted.get("inference", {}).pop("decision", None)
        persisted.get("test_evaluation", {}).pop("threshold_sweep", None)
        target = str(segmentation.get("target", "")).strip()
        mask_dirs = paths.get("mask_dirs", {})
        if target == "legacy":
            paths.pop("mask_dirs", None)
        elif target and isinstance(mask_dirs, dict) and target in mask_dirs:
            paths.pop("masks_dir", None)
            paths["mask_dirs"] = {target: mask_dirs[target]}

    model = persisted.get("model", {})
    model_name = str(model.get("name", "")).lower()
    unet_keys = {"decoder_normalization", "decoder_channels", "decoder_attention_type"}
    segformer_keys = {"encoder_depth", "decoder_segmentation_channels", "upsampling"}
    refinement_keys = {
        "encoder_depth",
        "decoder_segmentation_channels",
        "shallow_channels",
        "refine_half_channels",
        "refine_full_channels",
    }
    known_model_keys = unet_keys | segformer_keys | refinement_keys
    if model_name.startswith("unetplusplus"):
        _retain_known_relevant_keys(model, known_model_keys, unet_keys)
    elif model_name in {"segformer", "segformer_mit_b3", "segformer_mit_b5"}:
        _retain_known_relevant_keys(model, known_model_keys, segformer_keys)
    elif model_name in {
        "segformer_mit_b1_refinement",
        "segformer_mit_b2_refinement",
    }:
        _retain_known_relevant_keys(model, known_model_keys, refinement_keys)
    elif model_name in {"deeplabv3_resnet50", "fcn_resnet50"}:
        _retain_known_relevant_keys(model, known_model_keys | {"in_channels"}, set())

    loss = persisted.get("loss", {})
    loss_name = str(loss.get("name", "")).lower()
    loss_keys_by_name = {
        "bce_with_logits": set(),
        "bce": set(),
        "binary_cross_entropy_with_logits": set(),
        "bce_dice": {"bce_weight", "dice_weight", "smooth"},
        "bce_dice_cldice": {
            "bce_weight", "dice_weight", "soft_cldice_weight", "iterations",
            "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
        "bce_dice_soft_cldice": {
            "bce_weight", "dice_weight", "soft_cldice_weight", "iterations",
            "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
        "bcedicecldice": {
            "bce_weight", "dice_weight", "soft_cldice_weight", "iterations",
            "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
        "multiclass_ce_dice_loci_cldice": {
            "cross_entropy_weight", "dice_weight", "loci_cldice_weight",
            "iterations", "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
        "multiclass_geometry_ce_dice_loci_cldice": {
            "geometry_aware_ce_weight", "dice_weight", "soft_cldice_weight",
            "geometry_aware_ce", "iterations", "iterations_csv", "static_patch_iterations",
            "smooth", "cldice_smooth",
        },
        "tversky": {"alpha", "beta", "smooth"},
        "cldice": {"iterations", "iterations_csv", "static_patch_iterations", "cldice_smooth"},
        "soft_cldice": {"iterations", "iterations_csv", "static_patch_iterations", "cldice_smooth"},
        "softcldice": {"iterations", "iterations_csv", "static_patch_iterations", "cldice_smooth"},
        "tversky_soft_cldice": {
            "alpha", "beta", "tversky_weight", "soft_cldice_weight",
            "iterations", "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
        "tversky_softcldice": {
            "alpha", "beta", "tversky_weight", "soft_cldice_weight",
            "iterations", "iterations_csv", "static_patch_iterations", "smooth", "cldice_smooth",
        },
    }
    if loss_name in loss_keys_by_name:
        known_loss_keys = set().union(*loss_keys_by_name.values()) | {"threshold"}
        _retain_known_relevant_keys(
            loss,
            known_loss_keys,
            loss_keys_by_name[loss_name],
        )

    training_cache_enabled = bool(
        persisted.get("data", {}).get("train_patch_cache", {}).get(
            "enabled", True
        )
    )
    validation_cache_enabled = bool(
        persisted.get("validation", {})
        .get("full_image", {})
        .get("patch_cache", {})
        .get("enabled", False)
    )
    if (
        not training_cache_enabled
        and not validation_cache_enabled
    ) or loss.get("iterations_csv"):
        loss.pop("static_patch_iterations", None)

    scheduler = persisted.get("scheduler", {})
    scheduler_name = str(scheduler.get("name", "")).lower()
    known_scheduler_keys = {
        "mode", "factor", "patience", "threshold", "threshold_mode", "min_lr", "monitor"
    }
    if scheduler_name != "reduce_on_plateau":
        _retain_known_relevant_keys(scheduler, known_scheduler_keys, set())

    train = persisted.get("train", {})
    for legacy_key in (
        "enable_per_image_validation",
        "per_image_validation_interval",
        "full_image_validation",
        "full_image_monitor",
    ):
        train.pop(legacy_key, None)

    split = persisted.get("split", {})
    split_mode = str(split.get("mode", "")).lower()
    if split_mode not in {"kfold", "csv_kfold"}:
        persisted.pop("cv", None)
    if split_mode == "csv":
        split.pop("val_source_ids", None)
        split.pop("test_source_ids", None)
    elif split_mode == "train_val":
        split.pop("csv_path", None)
    elif split_mode == "csv_kfold":
        split.pop("val_source_ids", None)
        split.pop("test_source_ids", None)
    elif split_mode == "kfold":
        for key in ("csv_path", "val_source_ids", "test_source_ids"):
            split.pop(key, None)

    if training_date is not None:
        persisted.pop("training_date", None)
        return {"training_date": training_date, **persisted}
    return persisted


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
