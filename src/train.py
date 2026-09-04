from __future__ import annotations

import argparse
import atexit
import json
import statistics
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SegmentationPatchDataset, get_train_transforms
from src.data.patch_cache import (
    CachedSegmentationPatchDataset,
    build_epoch_training_crop_records,
    build_static_patch_cache,
    build_static_validation_patch_cache,
)
from src.data.discovery import discover_image_mask_pairs, discover_image_mask_sets
from src.data.fives import FivesPatchDataset, load_fives_training_records
from src.data.folds import (
    SplitDefinition,
    make_csv_kfold_splits,
    make_csv_train_val_test_split,
    make_grouped_kfold_splits,
    make_manual_train_val_split,
)
from src.data.sampling import patch_distribution
from src.data.soft_cldice_iterations import map_training_iterations_to_sources
from src.engine.trainer import Trainer, best_checkpoint_specs
from src.losses.factory import build_loss
from src.losses.geometry import build_geometry_weight_map_builder
from src.models.factory import build_model
from src.optim.factory import build_optimizer
from src.patching import build_original_image_records, build_patch_records
from src.schedulers.factory import build_scheduler
from src.utils.config import config_for_persistence, load_config, resolve_mask_dir
from src.utils.io import ensure_dir, save_csv, save_json, save_yaml
from src.utils.logging import setup_logger
from src.utils.seed import set_seed
from src.utils.run_resume import (
    append_resume_history, atomic_json, clean_incomplete_folds,
    contiguous_completed_folds, read_csv_rows, validate_completed_fold,
)


TORCH_SHARING_STRATEGY = "file_system"
SOFT_CLDICE_LOSS_NAMES = {
    "bce_dice_cldice",
    "bce_dice_soft_cldice",
    "bcedicecldice",
    "cldice",
    "soft_cldice",
    "softcldice",
    "tversky_soft_cldice",
    "tversky_softcldice",
    "multiclass_ce_dice_loci_cldice",
    "multiclass_geometry_ce_dice_loci_cldice",
}


def configure_torch_multiprocessing() -> str:
    torch.multiprocessing.set_sharing_strategy(TORCH_SHARING_STRATEGY)
    return torch.multiprocessing.get_sharing_strategy()


def _worker_init_fn(_: int) -> None:
    torch.multiprocessing.set_sharing_strategy(TORCH_SHARING_STRATEGY)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        _seed_dataset_transforms(
            worker_info.dataset,
            seed=int(torch.initial_seed() % (2**32)),
        )


def _seed_dataset_transforms(dataset: Dataset, seed: int) -> None:
    """Seed each distinct Albumentations pipeline in a worker deterministically."""
    pending = [dataset]
    seen_datasets: set[int] = set()
    seen_transforms: set[int] = set()
    transform_index = 0

    while pending:
        current = pending.pop()
        if id(current) in seen_datasets:
            continue
        seen_datasets.add(id(current))

        transforms = getattr(current, "transforms", None)
        if transforms is not None and id(transforms) not in seen_transforms:
            set_random_seed = getattr(transforms, "set_random_seed", None)
            if callable(set_random_seed):
                set_random_seed(int((seed + transform_index) % (2**32)))
                transform_index += 1
            seen_transforms.add(id(transforms))

        children = getattr(current, "datasets", None)
        if children is not None:
            pending.extend(children)


def _collect_optional_metric(values: list[float | None]) -> tuple[float | None, float | None]:
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return None, None
    mean_value = statistics.mean(valid_values)
    std_value = statistics.pstdev(valid_values) if len(valid_values) > 1 else 0.0
    return mean_value, std_value


def build_cross_fold_test_summary(fold_results: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    metric_names = sorted({
        key
        for result in fold_results
        for key, value in result.items()
        if key.startswith("mean_")
        and value is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    })
    fold_rows = [
        {
            "fold": int(result["fold"]),
            "checkpoint": result["checkpoint"],
            "output_dir": result["output_dir"],
            "num_test_images": int(result["num_test_images"]),
            "threshold": result["threshold"],
            **{metric_name: result.get(metric_name) for metric_name in metric_names},
        }
        for result in fold_results
    ]
    metric_rows: list[dict] = []
    metrics: dict[str, dict[str, float | int]] = {}
    for metric_name in metric_names:
        values = [
            float(result[metric_name])
            for result in fold_results
            if result.get(metric_name) is not None
        ]
        metric_summary = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "num_folds": len(values),
        }
        metrics[metric_name] = metric_summary
        metric_rows.append({"metric": metric_name, **metric_summary})

    payload = {
        "num_folds": len(fold_results),
        "num_test_images": (
            int(fold_results[0]["num_test_images"]) if fold_results else 0
        ),
        "folds": fold_rows,
        "metrics": metrics,
    }
    return payload, fold_rows, metric_rows


def persist_cross_fold_test_summary(
    output_dir: str | Path,
    fold_results: list[dict],
) -> dict:
    output_dir = ensure_dir(output_dir)
    payload, fold_rows, metric_rows = build_cross_fold_test_summary(fold_results)
    save_csv(output_dir / "fold_metrics.csv", fold_rows)
    save_csv(output_dir / "summary.csv", metric_rows)
    save_json(output_dir / "summary.json", payload)
    return payload


def build_checkpoint_test_comparison(
    checkpoint_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Build per-checkpoint rows and cross-fold monitor summaries."""
    metric_names = sorted({
        key
        for result in checkpoint_results
        for key, value in result.items()
        if key.startswith("mean_")
        and value is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    })
    rows = [
        {
            "fold": int(result["fold"]),
            "checkpoint_name": result["checkpoint_name"],
            "selection_monitor": result["selection_monitor"],
            "selection_mode": result["selection_mode"],
            "selection_epoch": int(result["selection_epoch"]),
            "selection_value": float(result["selection_value"]),
            "evaluation_id": result.get("evaluation_id"),
            "canonical_evaluated_checkpoint": result.get(
                "canonical_evaluated_checkpoint", result.get("checkpoint")
            ),
            "shared_evaluation": result.get("shared_evaluation", False),
            "matching_checkpoint_names": result.get(
                "matching_checkpoint_names", result["checkpoint_name"]
            ),
            "checkpoint": result["checkpoint"],
            "output_dir": result["output_dir"],
            "num_test_images": int(result["num_test_images"]),
            "num_join_images": result.get("num_join_images"),
            "threshold": result["threshold"],
            **{metric_name: result.get(metric_name) for metric_name in metric_names},
        }
        for result in checkpoint_results
    ]

    summary_rows: list[dict] = []
    checkpoint_names = list(dict.fromkeys(
        result["checkpoint_name"] for result in checkpoint_results
    ))
    for checkpoint_name in checkpoint_names:
        selected = [
            result
            for result in checkpoint_results
            if result["checkpoint_name"] == checkpoint_name
        ]
        summary_row = {
            "checkpoint_name": checkpoint_name,
            "selection_monitor": selected[0]["selection_monitor"],
            "selection_mode": selected[0]["selection_mode"],
            "num_folds": len(selected),
        }
        for metric_name in metric_names:
            values = [
                float(result[metric_name])
                for result in selected
                if result.get(metric_name) is not None
            ]
            summary_row[metric_name] = (statistics.mean(values) if values else None)
            summary_row[f"{metric_name}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            )
            summary_row[f"{metric_name}_num_folds"] = len(values)
        summary_rows.append(summary_row)
    return rows, summary_rows


def persist_checkpoint_test_comparison(
    output_dir: str | Path,
    checkpoint_results: list[dict],
    total_folds: int,
    *,
    persist_per_fold: bool = True,
) -> list[dict]:
    output_dir = ensure_dir(output_dir)
    rows, summary_rows = build_checkpoint_test_comparison(checkpoint_results)
    save_csv(output_dir / "checkpoint_comparison.csv", rows)
    save_csv(output_dir / "monitor_comparison_summary.csv", summary_rows)
    if total_folds > 1 and persist_per_fold:
        for fold in sorted({int(row["fold"]) for row in rows}):
            save_csv(
                output_dir / f"fold_{fold}" / "checkpoint_comparison.csv",
                [row for row in rows if int(row["fold"]) == fold],
            )
    return summary_rows


def checkpoint_selection_from_history(
    history: list[dict],
    monitor: str,
    mode: str,
) -> tuple[int, float]:
    candidates = [row for row in history if monitor in row]
    if not candidates:
        raise RuntimeError(f"No validation history is available for {monitor}.")
    selected = (
        max(candidates, key=lambda row: row[monitor])
        if mode == "max"
        else min(candidates, key=lambda row: row[monitor])
    )
    return int(selected["epoch"]), float(selected[monitor])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fungi segmentation with grouped cross-validation.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Path to the YAML config file (default: config.yaml).")
    source.add_argument("--resume-run", help="Existing run directory; its saved config.yaml is authoritative.")
    return parser.parse_args()


def _as_split_definition(split: SplitDefinition | tuple[list[str], list[str]]) -> SplitDefinition:
    if isinstance(split, SplitDefinition):
        return split
    train_sources, val_sources = split
    return SplitDefinition(
        train_sources=list(train_sources),
        val_sources=list(val_sources),
        test_sources=[],
    )


def build_splits(config: dict, original_records: list) -> tuple[list[SplitDefinition], str]:
    source_ids = [record.source_id for record in original_records]
    split_cfg = config.get("split", {})
    split_mode = str(split_cfg.get("mode", "train_val")).lower()

    if split_mode == "csv":
        split = make_csv_train_val_test_split(
            source_ids,
            csv_path=split_cfg.get("csv_path", "data/image_splits.csv"),
        )
        return [split], split_mode

    if split_mode == "csv_kfold":
        splits = make_csv_kfold_splits(
            source_ids,
            csv_path=split_cfg.get("csv_path", "data/image_splits.csv"),
            n_splits=int(config["cv"]["n_splits"]),
            shuffle_groups=bool(config["cv"]["shuffle_groups"]),
            random_state=int(config["cv"]["random_state"]),
        )
        return splits, split_mode

    if split_mode == "kfold":
        splits = make_grouped_kfold_splits(
            source_ids,
            n_splits=int(config["cv"]["n_splits"]),
            shuffle_groups=bool(config["cv"]["shuffle_groups"]),
            random_state=int(config["cv"]["random_state"]),
        )
        return [_as_split_definition(split) for split in splits], split_mode

    if split_mode == "train_val":
        splits = make_manual_train_val_split(
            source_ids,
            val_source_ids=split_cfg.get("val_source_ids", []),
        )
        split_definitions = [_as_split_definition(split) for split in splits]
        requested_test_sources = split_cfg.get("test_source_ids", [])
        if requested_test_sources:
            test_split = make_manual_train_val_split(source_ids, requested_test_sources)[0][1]
            split_definitions = [
                SplitDefinition(
                    train_sources=split.train_sources,
                    val_sources=split.val_sources,
                    test_sources=test_split,
                )
                for split in split_definitions
            ]
        return split_definitions, split_mode

    raise ValueError(
        f"Unsupported split mode: {split_mode}. Expected 'csv', 'csv_kfold', "
        "'train_val', or 'kfold'."
    )


def build_full_image_validation_patching_config(patching_config: dict) -> dict:
    """Build full-image validation settings with 50% patch overlap."""
    config = deepcopy(patching_config)
    overlap = int(config["patch_size"]) // 2
    config["overlap"] = overlap
    config["stride"] = overlap
    return config


def select_full_image_validation_records(records: list, validation_config: dict) -> list:
    """Select source images used by the slower stitched validation pass."""
    selection_config = validation_config.get("full_image", {})
    selection = str(selection_config.get("selection", "all")).lower()
    if selection == "all":
        return list(records)
    if selection != "smallest_area":
        raise ValueError(
            "validation.full_image.selection must be 'all' or 'smallest_area'."
        )

    max_images = selection_config.get("max_images")
    if max_images is None or int(max_images) <= 0:
        raise ValueError(
            "validation.full_image.max_images must be positive for smallest_area selection."
        )
    return sorted(
        records,
        key=lambda record: (record.width * record.height, record.source_id),
    )[: int(max_images)]


def split_manifest_rows(
    splits: list[SplitDefinition | tuple[list[str], list[str]]],
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for fold_index, split in enumerate(splits):
        split_definition = _as_split_definition(split)
        for source_id in split_definition.train_sources:
            rows.append({"fold": fold_index, "split": "train", "source_id": source_id})
        for source_id in split_definition.val_sources:
            rows.append({"fold": fold_index, "split": "val", "source_id": source_id})
        for source_id in split_definition.test_sources:
            rows.append({"fold": fold_index, "split": "test", "source_id": source_id})
    return rows


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["worker_init_fn"] = _worker_init_fn
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def combine_training_datasets(
    fungal_dataset: Dataset,
    fives_dataset: Dataset | None,
) -> Dataset:
    if fives_dataset is None:
        return fungal_dataset
    return ConcatDataset([fungal_dataset, fives_dataset])


def create_run_dir(runs_root: Path, project_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{project_name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{project_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def segmentation_summary_metadata(config: dict, mask_dir: Path) -> dict:
    segmentation = config.get("segmentation", {})
    segmentation_mode = str(segmentation.get("mode", "binary")).lower()
    if segmentation_mode == "multiclass":
        mask_dirs = {
            str(name): str(path)
            for name, path in config.get("paths", {}).get("mask_dirs", {}).items()
        }
        return {
            "segmentation_mode": "multiclass",
            "segmentation_target": None,
            "mask_dir": None,
            "mask_dirs": mask_dirs,
        }

    target = str(segmentation.get("target", "legacy"))
    return {
        "segmentation_mode": "binary",
        "segmentation_target": target,
        "mask_dir": str(mask_dir),
        "mask_dirs": {target: str(mask_dir)},
    }


def log_run_summary(
    logger,
    config: dict,
    device: torch.device,
    num_images: int,
    split_mode: str,
    mask_dir: Path,
) -> None:
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    patching_cfg = config["patching"]
    optimizer_cfg = config["optimizer"]
    segmentation_metadata = segmentation_summary_metadata(config, mask_dir)
    mask_summary = ", ".join(
        f"{name}:{path}"
        for name, path in segmentation_metadata["mask_dirs"].items()
    )
    if "encoder_lr" in optimizer_cfg and "decoder_lr" in optimizer_cfg:
        learning_rate_summary = (
            f"encoder_lr={optimizer_cfg['encoder_lr']} | "
            f"decoder_lr={optimizer_cfg['decoder_lr']}"
        )
    else:
        learning_rate_summary = f"lr={optimizer_cfg['lr']}"

    logger.info("Training summary")
    logger.info(
        "Model: %s | encoder=%s | epochs=%s | batch_size=%s | optimizer=%s | %s | device=%s",
        model_cfg["name"],
        model_cfg.get("encoder_name", "-"),
        train_cfg["epochs"],
        data_cfg["batch_size"],
        optimizer_cfg["name"],
        learning_rate_summary,
        device,
    )
    logger.info(
        "Dataset: %s images | mode=%s | target=%s | masks=%s | split_mode=%s | patch_size=%s | stride=%s | empty_patch_filter=%s",
        num_images,
        segmentation_metadata["segmentation_mode"],
        segmentation_metadata["segmentation_target"] or "-",
        mask_summary,
        split_mode,
        patching_cfg["patch_size"],
        patching_cfg["stride"],
        patching_cfg["filter_empty_patches"],
    )
    logger.info(
        "Loader: num_workers=%s | persistent_workers=%s | prefetch_factor=%s | pin_memory=%s",
        data_cfg["num_workers"],
        data_cfg.get("persistent_workers", False),
        data_cfg.get("prefetch_factor"),
        data_cfg["pin_memory"],
    )


def log_fold_summary(
    logger,
    fold_index: int,
    total_folds: int,
    split_mode: str,
    train_originals: list,
    val_originals: list,
    test_originals: list,
    train_patch_records: list,
    test_patch_records: list,
) -> None:
    split_label = "Fold" if split_mode == "kfold" else "Split"
    logger.info(
        "%s %s/%s | train_images=%s | val_images=%s | test_images=%s | train_patches=%s | test_patches=%s",
        split_label,
        fold_index + 1,
        total_folds,
        len(train_originals),
        len(val_originals),
        len(test_originals),
        len(train_patch_records),
        len(test_patch_records),
    )
    # logger.info(
    #     "Fold %s split | train_sources=%s | val_sources=%s",
    #     fold_index + 1,
    #     ", ".join(record.source_id for record in train_originals),
    #     ", ".join(record.source_id for record in val_originals),
    # )


def main() -> None:
    args = parse_args()
    resume_run = Path(args.resume_run).resolve() if args.resume_run else None
    config_path = resume_run / "config.yaml" if resume_run else Path(args.config or "config.yaml")
    if resume_run is not None and not config_path.is_file():
        raise FileNotFoundError(f"Resume run has no saved config: {config_path}")
    config = load_config(config_path)
    training_date = datetime.now().astimezone().date().isoformat()

    set_seed(int(config["train"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "auto")))

    project_name = config["project"]["name"]
    runs_root = ensure_dir(Path(config["paths"]["runs_dir"]))
    run_dir = resume_run if resume_run is not None else create_run_dir(runs_root, project_name)
    outputs_root = ensure_dir(Path(config["paths"]["outputs_dir"]) / project_name)
    logger = setup_logger("train", run_dir / "logs")
    if resume_run is None:
        save_yaml(
            run_dir / "config.yaml",
            config_for_persistence(config, training_date=training_date),
        )
    sharing_strategy = configure_torch_multiprocessing()
    logger.info("Using device: %s", device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Torch multiprocessing sharing strategy: %s", sharing_strategy)

    segmentation_mode = str(config.get("segmentation", {}).get("mode", "binary")).lower()
    join_masks_cfg = config.get("join_masks", {})
    optional_mask_dirs = (
        {"join": join_masks_cfg["masks_dir"]}
        if segmentation_mode == "multiclass" and join_masks_cfg.get("enabled", False)
        else None
    )
    merge_join_masks = bool(join_masks_cfg.get("merge_with_loci", False))
    if segmentation_mode == "multiclass":
        mask_dir = Path(config["paths"]["mask_dirs"]["loci"])
        pairs, diagnostics = discover_image_mask_sets(
            config["paths"]["images_dir"],
            {
                "loci": config["paths"]["mask_dirs"]["loci"],
                "inoculum": config["paths"]["mask_dirs"]["inoculum"],
            },
            config["data"]["image_extensions"],
            optional_mask_dirs=optional_mask_dirs,
        )
    else:
        mask_dir = resolve_mask_dir(config)
        pairs, diagnostics = discover_image_mask_pairs(
            config["paths"]["images_dir"],
            mask_dir,
            config["data"]["image_extensions"],
        )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found.")
    if segmentation_mode == "multiclass":
        for class_name, stems in diagnostics["missing_masks"].items():
            if stems:
                logger.warning("Missing %s masks for %s images: %s", class_name, len(stems), ", ".join(stems))
        for class_name, stems in diagnostics["missing_images"].items():
            if stems:
                logger.warning("Found %s %s masks without matching images.", len(stems), class_name)
        if diagnostics.get("dimension_mismatches"):
            logger.warning(
                "Excluded %s image/mask sets with dimension mismatches.",
                len(diagnostics["dimension_mismatches"]),
            )
        if diagnostics.get("optional_dimension_mismatches"):
            logger.warning(
                "Ignored %s optional join masks with dimension mismatches.",
                len(diagnostics["optional_dimension_mismatches"]),
            )
    else:
        if diagnostics["missing_masks"]:
            logger.warning("Missing masks for %s images.", len(diagnostics["missing_masks"]))
        if diagnostics["missing_images"]:
            logger.warning("Found %s masks without matching images.", len(diagnostics["missing_images"]))

    original_records = build_original_image_records(pairs)
    iterations_csv = config["loss"].get("iterations_csv")
    if (
        iterations_csv
        and str(config["loss"]["name"]).lower() not in SOFT_CLDICE_LOSS_NAMES
    ):
        raise ValueError(
            "loss.iterations_csv requires a loss that contains Soft-clDice."
        )
    soft_cldice_iterations = (
        map_training_iterations_to_sources(iterations_csv, original_records)
        if iterations_csv
        else None
    )
    fives_patch_records = load_fives_training_records(config)
    splits, split_mode = build_splits(config, original_records)
    manifest_rows = split_manifest_rows(splits)
    if resume_run is not None:
        manifest_path = run_dir / "split_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without split manifest: {manifest_path}")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("splits", [])
        if saved_manifest != manifest_rows:
            raise RuntimeError("Saved split manifest does not match rediscovered data and recomputed splits.")
    else:
        save_csv(run_dir / "split_manifest.csv", manifest_rows)
        save_json(run_dir / "split_manifest.json", {"splits": manifest_rows})
    log_run_summary(
        logger,
        config,
        device,
        num_images=len(original_records),
        split_mode=split_mode,
        mask_dir=mask_dir,
    )
    if fives_patch_records:
        logger.info(
            "FIVES auxiliary training enabled | images=%s | patches_per_image=4 | total_patches=%s",
            len({record.source_id for record in fives_patch_records}),
            len(fives_patch_records),
        )
    if soft_cldice_iterations is not None:
        logger.info(
            "Per-image Soft-clDice iterations enabled | csv=%s | images=%s | "
            "minimum=%s | maximum=%s",
            iterations_csv,
            len(soft_cldice_iterations),
            min(soft_cldice_iterations.values()),
            max(soft_cldice_iterations.values()),
        )
        if fives_patch_records:
            logger.info(
                "FIVES patches use the fixed fallback loss.iterations=%s.",
                config["loss"]["iterations"],
            )

    fold_results = []
    fold_test_results: list[dict] = []
    checkpoint_test_results: list[dict] = []
    checkpoint_comparison_summary: list[dict] = []
    configured_checkpoint_specs = best_checkpoint_specs(config["checkpointing"])
    primary_checkpoint_name = str(
        config["checkpointing"]["selections"][
            config["checkpointing"]["primary"]
        ]["filename"]
    )
    cross_fold_test_summary: dict | None = None
    all_epoch_rows: list[dict[str, float]] = []
    data_cfg = config["data"]
    patching_cfg = {
        **config["patching"],
        "include_join_masks": merge_join_masks,
    }
    validation_cfg = config["validation"]
    full_image_validation_patching_cfg = build_full_image_validation_patching_config(
        patching_cfg
    )
    augmentations_cfg = config.get("augmentations", {})
    target_weight_builder = build_geometry_weight_map_builder(config["loss"])
    total_folds = len(splits)
    completed_folds: list[int] = []
    first_incomplete = 0
    if resume_run is not None:
        existing_fold_rows = read_csv_rows(run_dir / "fold_metrics.csv")
        completed_folds = contiguous_completed_folds(existing_fold_rows, total_folds)
        test_required = bool(config.get("test_evaluation", {}).get("enabled", True))
        for completed_fold in completed_folds:
            validate_completed_fold(
                run_dir,
                completed_fold,
                test_required,
                [filename for filename, _, _, _ in configured_checkpoint_specs],
            )
        first_incomplete = len(completed_folds)
        removed = clean_incomplete_folds(run_dir, first_incomplete, total_folds)
        # Preserve only completed-fold rows in run-level incremental artifacts.
        save_csv(run_dir / "fold_metrics.csv", [row for row in existing_fold_rows if int(row["fold"]) < first_incomplete])
        existing_epoch_rows = read_csv_rows(run_dir / "epoch_metrics.csv")
        save_csv(run_dir / "epoch_metrics.csv", [row for row in existing_epoch_rows if int(row["fold"]) < first_incomplete])
        comparison_path = run_dir / "test-evaluation" / "checkpoint_comparison.csv"
        old_comparisons = read_csv_rows(comparison_path)
        if old_comparisons:
            save_csv(comparison_path, [row for row in old_comparisons if int(row["fold"]) < first_incomplete])
        append_resume_history(run_dir, {
            "first_incomplete_fold": first_incomplete,
            "completed_folds": completed_folds,
            "removed_partial_artifacts": removed,
            "note": "Incomplete fold restarts at epoch 1 with current code and saved config.",
        })
        if first_incomplete >= total_folds:
            atomic_json(run_dir / "run_state.json", {
                "total_folds": total_folds, "completed_folds": completed_folds,
                "active_fold": None, "attempt_count": 1, "status": "complete",
            })
            logger.info("All %s folds are already complete; nothing to resume.", total_folds)
            return
    state_path = run_dir / "run_state.json"
    previous_attempts = 0
    if state_path.is_file():
        previous_attempts = int(json.loads(state_path.read_text(encoding="utf-8")).get("attempt_count", 0))
    attempt_count = previous_attempts + 1
    atomic_json(state_path, {
        "total_folds": total_folds, "completed_folds": completed_folds,
        "active_fold": first_incomplete if first_incomplete < total_folds else None,
        "attempt_count": attempt_count, "status": "running",
    })

    if resume_run is not None:
        # Reconstruct aggregation state without changing completed fold artifacts.
        for row in read_csv_rows(run_dir / "fold_metrics.csv"):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value) if key != "fold" and value != "" else (int(value) if key == "fold" else None)
                except (TypeError, ValueError):
                    converted[key] = value
            fold_results.append(converted)
        all_epoch_rows.extend(read_csv_rows(run_dir / "epoch_metrics.csv"))
        for raw_row in read_csv_rows(run_dir / "test-evaluation" / "checkpoint_comparison.csv"):
            row = dict(raw_row)
            for key in ("fold", "selection_epoch", "num_test_images", "num_join_images"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            for key, value in list(row.items()):
                if (key == "selection_value" or key.startswith("mean_")) and value not in (None, ""):
                    row[key] = float(value)
            checkpoint_test_results.append(row)
        for row in checkpoint_test_results:
            if row.get("checkpoint_name") != primary_checkpoint_name:
                continue
            converted = {
                key: value for key, value in row.items()
                if key.startswith("mean_") or key in {
                    "fold", "checkpoint", "output_dir", "threshold",
                    "num_test_images", "num_join_images",
                }
            }
            fold_test_results.append(converted)
        if checkpoint_test_results:
            checkpoint_comparison_summary = persist_checkpoint_test_comparison(
                run_dir / "test-evaluation", checkpoint_test_results, total_folds,
                persist_per_fold=False,
            )
        if fold_test_results and total_folds > 1:
            cross_fold_test_summary = persist_cross_fold_test_summary(
                run_dir / "test-evaluation", fold_test_results
            )

    cache_enabled = bool(data_cfg.get("train_patch_cache", {}).get("enabled", True))
    static_iteration_cfg = config["loss"].get("static_patch_iterations", {})
    compute_static_iterations = (
        cache_enabled
        and soft_cldice_iterations is None
        and str(config["loss"]["name"]).lower() in SOFT_CLDICE_LOSS_NAMES
        and bool(static_iteration_cfg.get("enabled", True))
    )
    static_cache = None
    if cache_enabled:
        training_source_ids = {
            source_id for split in splits for source_id in split.train_sources
        }
        cache_originals = [
            record for record in original_records
            if record.source_id in training_source_ids
        ]
        static_cache = build_static_patch_cache(
            cache_originals,
            run_dir,
            patching_cfg,
            segmentation_mode=segmentation_mode,
            merge_join_masks=merge_join_masks,
            compute_soft_cldice_iterations=compute_static_iterations,
            iteration_margin=int(static_iteration_cfg.get("margin_iterations", 10)),
            iteration_round_up_to=int(static_iteration_cfg.get("round_up_to", 10)),
        )
        atexit.register(static_cache.cleanup)
        iteration_values = [
            record.soft_cldice_iterations for record in static_cache.records
            if record.soft_cldice_iterations is not None
        ]
        logger.info(
            "Static training cache ready | regions=%s | region_size=%sx%s | "
            "sources=%s | soft_cldice_iterations=%s",
            len(static_cache.records),
            int(patching_cfg["patch_size"]) + int(patching_cfg["overlap"]),
            int(patching_cfg["patch_size"]) + int(patching_cfg["overlap"]),
            len(cache_originals),
            (
                f"{min(iteration_values)}-{max(iteration_values)}"
                if iteration_values else "fixed-or-csv"
            ),
        )

    for fold_index, split in enumerate(splits):
        if fold_index in completed_folds:
            logger.info("Skipping completed fold %s", fold_index)
            continue
        atomic_json(state_path, {
            "total_folds": total_folds, "completed_folds": completed_folds,
            "active_fold": fold_index, "attempt_count": attempt_count, "status": "running",
        })
        logger.info("Preparing fold %s", fold_index)
        set_seed(int(config["train"]["seed"]) + fold_index)
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        tensorboard_dir = ensure_dir(fold_dir / "tensorboard")
        train_sources = split.train_sources
        val_sources = split.val_sources
        test_sources = split.test_sources

        train_originals = [record for record in original_records if record.source_id in set(train_sources)]
        val_originals = [record for record in original_records if record.source_id in set(val_sources)]
        test_originals = [record for record in original_records if record.source_id in set(test_sources)]
        full_image_enabled = bool(validation_cfg["full_image"]["enabled"])
        full_image_val_originals = select_full_image_validation_records(
            val_originals, validation_cfg
        )
        validation_patch_cache_cfg = validation_cfg["full_image"].get(
            "patch_cache", {}
        )
        validation_patch_cache_enabled = (
            full_image_enabled
            and bool(validation_patch_cache_cfg.get("enabled", False))
        )
        compute_validation_static_iterations = (
            validation_patch_cache_enabled
            and soft_cldice_iterations is None
            and str(config["loss"]["name"]).lower() in SOFT_CLDICE_LOSS_NAMES
            and bool(static_iteration_cfg.get("enabled", True))
        )
        validation_patch_cache = None
        validation_iteration_values: list[int] = []
        if validation_patch_cache_enabled:
            cache_started = time.perf_counter()
            validation_patch_cache = build_static_validation_patch_cache(
                full_image_val_originals,
                fold_dir,
                full_image_validation_patching_cfg,
                segmentation_mode=segmentation_mode,
                merge_join_masks=merge_join_masks,
                compute_soft_cldice_iterations=(
                    compute_validation_static_iterations
                ),
                iteration_margin=int(
                    static_iteration_cfg.get("margin_iterations", 10)
                ),
                iteration_round_up_to=int(
                    static_iteration_cfg.get("round_up_to", 10)
                ),
            )
            atexit.register(validation_patch_cache.cleanup)
            validation_iteration_values = [
                record.soft_cldice_iterations
                for record in validation_patch_cache.records
                if record.soft_cldice_iterations is not None
            ]
            logger.info(
                "Static validation cache ready | fold=%s | patches=%s | "
                "patch_size=%sx%s | sources=%s | build_seconds=%.3f | "
                "soft_cldice_iterations=%s",
                fold_index,
                len(validation_patch_cache.records),
                int(full_image_validation_patching_cfg["patch_size"]),
                int(full_image_validation_patching_cfg["patch_size"]),
                len(validation_patch_cache.sources),
                time.perf_counter() - cache_started,
                (
                    f"{min(validation_iteration_values)}-"
                    f"{max(validation_iteration_values)}"
                    if validation_iteration_values else "fixed-or-csv"
                ),
            )

        if static_cache is not None:
            train_patch_records = build_epoch_training_crop_records(
                static_cache,
                train_sources,
                patching_cfg,
                epoch=1,
                base_seed=int(config["train"]["seed"]),
                fold_index=fold_index,
                segmentation_mode=segmentation_mode,
                merge_join_masks=merge_join_masks,
            )
        else:
            train_patch_records = build_patch_records(
                train_originals,
                patching_cfg,
                phase="train",
                epoch=1,
                base_seed=int(config["train"]["seed"]),
            )
        combined_train_patch_records = train_patch_records + fives_patch_records
        test_patch_records = build_patch_records(
            test_originals,
            patching_cfg,
            phase="validation",
        )
        log_fold_summary(
            logger=logger,
            fold_index=fold_index,
            total_folds=total_folds,
            split_mode=split_mode,
            train_originals=train_originals,
            val_originals=val_originals,
            test_originals=test_originals,
            train_patch_records=combined_train_patch_records,
            test_patch_records=test_patch_records,
        )
        logger.info(
            "Validation start_epoch=%s | full_image_enabled=%s | full_image_batch_size=%s | selection=%s | images=%s/%s | sources=%s",
            validation_cfg["start_epoch"],
            full_image_enabled,
            validation_cfg["full_image"]["batch_size"],
            validation_cfg["full_image"]["selection"],
            len(full_image_val_originals),
            len(val_originals),
            ", ".join(record.source_id for record in full_image_val_originals),
        )

        train_transforms = get_train_transforms(
            data_cfg.get("image_size"),
            augmentations_config=augmentations_cfg,
            seed=int(config["train"]["seed"]),
        )
        train_dataset = SegmentationPatchDataset(
            records=train_patch_records,
            mask_threshold=int(patching_cfg["mask_threshold"]),
            transforms=train_transforms,
            image_resampling=str(patching_cfg.get("image_resampling", "lanczos")),
            mask_resampling=str(patching_cfg.get("mask_resampling", "foreground_preserving")),
            segmentation_mode=segmentation_mode,
            target_weight_builder=target_weight_builder,
            merge_join_masks=merge_join_masks,
            soft_cldice_iterations=soft_cldice_iterations,
            default_soft_cldice_iterations=int(config["loss"]["iterations"]),
        )
        fives_dataset = (
            FivesPatchDataset(
                records=fives_patch_records,
                mask_threshold=int(patching_cfg["mask_threshold"]),
                transforms=train_transforms,
                segmentation_mode=segmentation_mode,
                target_weight_builder=target_weight_builder,
                soft_cldice_iterations=(
                    {}
                    if soft_cldice_iterations is not None or compute_static_iterations
                    else None
                ),
                default_soft_cldice_iterations=int(config["loss"]["iterations"]),
            )
            if fives_patch_records
            else None
        )

        fold_static_records = (
            [
                record for record in static_cache.records
                if record.source_id in set(train_sources)
            ]
            if static_cache is not None
            else []
        )
        fold_iteration_values = [
            record.soft_cldice_iterations for record in fold_static_records
            if record.soft_cldice_iterations is not None
        ]
        patch_diagnostics = {
            "train": patch_distribution(combined_train_patch_records),
            "static_cache": {
                "enabled": static_cache is not None,
                "regions": len(fold_static_records),
                "cache_size": (
                    int(patching_cfg["patch_size"]) + int(patching_cfg["overlap"])
                    if static_cache is not None else None
                ),
                "automatic_soft_cldice_iterations": compute_static_iterations,
                "minimum_soft_cldice_iterations": (
                    min(fold_iteration_values) if fold_iteration_values else None
                ),
                "maximum_soft_cldice_iterations": (
                    max(fold_iteration_values) if fold_iteration_values else None
                ),
            },
            "fives": patch_distribution(fives_patch_records),
            "test": patch_distribution(test_patch_records),
            "full_image_validation": {
                "enabled": full_image_enabled,
                "start_epoch": validation_cfg["start_epoch"],
                "batch_size": validation_cfg["full_image"]["batch_size"],
                "interval_epochs": validation_cfg["full_image"]["interval_epochs"],
                "selection": validation_cfg["full_image"]["selection"],
                "max_images": validation_cfg["full_image"]["max_images"],
                "patch_cache": {
                    "enabled": validation_patch_cache is not None,
                    "patches": (
                        len(validation_patch_cache.records)
                        if validation_patch_cache is not None else 0
                    ),
                    "automatic_soft_cldice_iterations": (
                        compute_validation_static_iterations
                    ),
                    "minimum_soft_cldice_iterations": (
                        min(validation_iteration_values)
                        if validation_iteration_values else None
                    ),
                    "maximum_soft_cldice_iterations": (
                        max(validation_iteration_values)
                        if validation_iteration_values else None
                    ),
                },
                "sources": [
                    {
                        "source_id": record.source_id,
                        "width": record.width,
                        "height": record.height,
                        "area": record.width * record.height,
                    }
                    for record in full_image_val_originals
                ],
            },
        }
        save_json(fold_dir / "patch_distribution.json", patch_diagnostics)

        def make_train_loader_for_epoch(epoch: int) -> DataLoader:
            if static_cache is not None:
                epoch_records = build_epoch_training_crop_records(
                    static_cache,
                    train_sources,
                    patching_cfg,
                    epoch=epoch,
                    base_seed=int(config["train"]["seed"]),
                    fold_index=fold_index,
                    segmentation_mode=segmentation_mode,
                    merge_join_masks=merge_join_masks,
                )
                fungal_epoch_dataset = CachedSegmentationPatchDataset(
                    epoch_records,
                    static_cache,
                    train_transforms,
                    segmentation_mode,
                    int(patching_cfg["mask_threshold"]),
                    merge_join_masks,
                    target_weight_builder,
                    soft_cldice_iterations,
                    int(config["loss"]["iterations"]),
                )
            else:
                epoch_records = build_patch_records(
                    train_originals,
                    patching_cfg,
                    phase="train",
                    epoch=epoch,
                    base_seed=int(config["train"]["seed"]),
                )
                train_dataset.set_records(epoch_records)
                fungal_epoch_dataset = train_dataset
            return make_loader(
                combine_training_datasets(fungal_epoch_dataset, fives_dataset),
                batch_size=int(data_cfg["batch_size"]),
                num_workers=int(data_cfg["num_workers"]),
                pin_memory=bool(data_cfg["pin_memory"]),
                shuffle=True,
                persistent_workers=bool(data_cfg.get("persistent_workers", False)),
                prefetch_factor=(
                    int(data_cfg["prefetch_factor"])
                    if data_cfg.get("prefetch_factor") is not None
                    else None
                ),
            )


        model = build_model(config["model"]).to(device)
        loss_fn = build_loss(config["loss"])
        optimizer = build_optimizer(model, config["optimizer"])
        scheduler = build_scheduler(optimizer, config["scheduler"])
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

        def persist_run_epoch_metrics(epoch_metrics: dict[str, float]) -> None:
            all_epoch_rows.append({"fold": fold_index, **epoch_metrics})
            save_csv(run_dir / "epoch_metrics.csv", all_epoch_rows)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_config={
                **config["train"],
                "scheduler_monitor": config["scheduler"].get(
                    "monitor", config["checkpointing"]["selections"][
                        config["checkpointing"]["primary"]
                    ]["monitor"]
                ),
            },
            loss_config=config["loss"],
            logger=logger,
            fold_dir=Path(fold_dir),
            data_config={**data_cfg, **full_image_validation_patching_cfg},
            augmentations_config=augmentations_cfg,
            val_original_records=full_image_val_originals,
            tensorboard_writer=writer,
            fold_index=fold_index,
            segmentation_config=config.get("segmentation", {}),
            join_masks_config=join_masks_cfg,
            validation_config=validation_cfg,
            checkpointing_config=config["checkpointing"],
            target_weight_builder=target_weight_builder,
            soft_cldice_iterations=soft_cldice_iterations,
            default_soft_cldice_iterations=int(config["loss"]["iterations"]),
            validation_patch_cache=validation_patch_cache,
            epoch_metrics_callback=persist_run_epoch_metrics,
        )
        fold_result = trainer.fit(
            None,
            epochs=int(config["train"]["epochs"]),
            train_loader_factory=make_train_loader_for_epoch,
        )
        if validation_patch_cache is not None:
            atexit.unregister(validation_patch_cache.cleanup)
        writer.close()
        if test_originals and bool(config.get("test_evaluation", {}).get("enabled", True)):
            from src.inference.test_evaluation import run_test_evaluation

            evaluation_root = run_dir / "test-evaluation"
            fold_evaluation_root = (
                evaluation_root / f"fold_{fold_index}"
                if total_folds > 1
                else evaluation_root
            )
            selections: list[dict] = []
            for checkpoint_name, monitor, mode, _ in configured_checkpoint_specs:
                checkpoint_path = fold_dir / checkpoint_name
                if not checkpoint_path.is_file():
                    raise RuntimeError(f"Expected validation checkpoint was not saved: {checkpoint_path}")
                selected_epoch, selected_value = checkpoint_selection_from_history(
                    fold_result["history"], monitor, mode
                )
                selections.append({
                    "checkpoint_name": checkpoint_name, "checkpoint_path": checkpoint_path,
                    "selection_monitor": monitor, "selection_mode": mode,
                    "selection_epoch": selected_epoch, "selection_value": selected_value,
                })

            by_epoch: dict[int, list[dict]] = {}
            for selection in selections:
                by_epoch.setdefault(selection["selection_epoch"], []).append(selection)
            for selected_epoch, matching in by_epoch.items():
                canonical = matching[0]
                evaluation_id = f"fold_{fold_index}_epoch_{selected_epoch}"
                evaluation_dir = fold_evaluation_root / f"epoch_{selected_epoch}"
                test_result = run_test_evaluation(
                    canonical["checkpoint_path"], config, evaluation_dir, device
                )
                matching_names = [item["checkpoint_name"] for item in matching]
                for selection in matching:
                    comparison_result = {
                        "fold": fold_index,
                        **{key: value for key, value in selection.items() if key != "checkpoint_path"},
                        "evaluation_id": evaluation_id,
                        "canonical_evaluated_checkpoint": str(canonical["checkpoint_path"]),
                        "shared_evaluation": len(matching) > 1,
                        "matching_checkpoint_names": ";".join(matching_names),
                        **test_result,
                    }
                    checkpoint_test_results.append(comparison_result)
                    if selection["checkpoint_name"] == primary_checkpoint_name:
                        fold_result.update({
                            "test_dice_per_image": test_result["mean_dice"],
                            "test_iou_per_image": test_result["mean_iou"],
                        })
                        if total_folds > 1:
                            fold_test_results.append({"fold": fold_index, **test_result})
                            cross_fold_test_summary = persist_cross_fold_test_summary(
                                evaluation_root, fold_test_results
                            )
                checkpoint_comparison_summary = persist_checkpoint_test_comparison(
                    evaluation_root, checkpoint_test_results, total_folds
                )
                logger.info(
                    "Fold %s epoch %s test evaluation shared by %s - output=%s",
                    fold_index, selected_epoch, ", ".join(matching_names), evaluation_dir,
                )
        fold_result["fold"] = fold_index
        fold_result["num_train_patches"] = len(combined_train_patch_records)
        fold_result["num_train_fives_patches"] = len(fives_patch_records)
        fold_result["num_test_patches"] = len(test_patch_records)
        fold_result["num_train_normal_patches"] = sum(
            1 for record in combined_train_patch_records if record.scale_label == "normal"
        )
        fold_result["num_train_scaled_context_patches"] = sum(
            1 for record in combined_train_patch_records if record.scale_label == "scaled_context"
        )
        fold_result["num_test_normal_patches"] = sum(
            1 for record in test_patch_records if record.scale_label == "normal"
        )
        fold_results.append(fold_result)
        save_csv(
            run_dir / "fold_metrics.csv",
            [{key: value for key, value in item.items() if key != "history"} for item in fold_results],
        )
        completed_folds.append(fold_index)
        atomic_json(state_path, {
            "total_folds": total_folds, "completed_folds": completed_folds,
            "active_fold": None, "attempt_count": attempt_count,
            "status": "complete" if len(completed_folds) == total_folds else "running",
        })

    val_dice_per_image_mean, val_dice_per_image_std = _collect_optional_metric(
        [item.get("val_dice_per_image") for item in fold_results]
    )
    val_iou_per_image_mean, val_iou_per_image_std = _collect_optional_metric(
        [item.get("val_iou_per_image") for item in fold_results]
    )
    val_cldice_per_image_mean, val_cldice_per_image_std = _collect_optional_metric(
        [item.get("val_cldice_per_image") for item in fold_results]
    )
    val_dice_cldice_per_image_mean, val_dice_cldice_per_image_std = _collect_optional_metric(
        [item.get("val_dice_cldice_per_image") for item in fold_results]
    )
    test_dice_per_patch_mean, test_dice_per_patch_std = _collect_optional_metric(
        [item.get("test_dice_per_patch") for item in fold_results]
    )
    test_iou_per_patch_mean, test_iou_per_patch_std = _collect_optional_metric(
        [item.get("test_iou_per_patch") for item in fold_results]
    )
    test_dice_macro_resolution_mean, test_dice_macro_resolution_std = _collect_optional_metric(
        [item.get("test_dice_macro_resolution") for item in fold_results]
    )
    test_iou_macro_resolution_mean, test_iou_macro_resolution_std = _collect_optional_metric(
        [item.get("test_iou_macro_resolution") for item in fold_results]
    )
    test_dice_per_image_mean, test_dice_per_image_std = _collect_optional_metric(
        [item.get("test_dice_per_image") for item in fold_results]
    )
    test_iou_per_image_mean, test_iou_per_image_std = _collect_optional_metric(
        [item.get("test_iou_per_image") for item in fold_results]
    )
    summary = {
        "project": project_name,
        "run_dir": str(run_dir),
        "split_mode": split_mode,
        **segmentation_summary_metadata(config, mask_dir),
        "folds": fold_results,
        "test_evaluation": cross_fold_test_summary,
        "checkpoint_test_comparison": checkpoint_comparison_summary,
        "mean_dice_per_image": val_dice_per_image_mean,
        "std_dice_per_image": val_dice_per_image_std,
        "mean_iou_per_image": val_iou_per_image_mean,
        "std_iou_per_image": val_iou_per_image_std,
        "mean_cldice_per_image": val_cldice_per_image_mean,
        "std_cldice_per_image": val_cldice_per_image_std,
        "mean_dice_cldice_per_image": val_dice_cldice_per_image_mean,
        "std_dice_cldice_per_image": val_dice_cldice_per_image_std,
        "mean_test_dice_per_patch": test_dice_per_patch_mean,
        "std_test_dice_per_patch": test_dice_per_patch_std,
        "mean_test_iou_per_patch": test_iou_per_patch_mean,
        "std_test_iou_per_patch": test_iou_per_patch_std,
        "mean_test_dice_macro_resolution": test_dice_macro_resolution_mean,
        "std_test_dice_macro_resolution": test_dice_macro_resolution_std,
        "mean_test_iou_macro_resolution": test_iou_macro_resolution_mean,
        "std_test_iou_macro_resolution": test_iou_macro_resolution_std,
        "mean_test_dice_per_image": test_dice_per_image_mean,
        "std_test_dice_per_image": test_dice_per_image_std,
        "mean_test_iou_per_image": test_iou_per_image_mean,
        "std_test_iou_per_image": test_iou_per_image_std,
        "num_original_images": len(original_records),
    }
    save_json(run_dir / "cv_summary.json", summary)
    save_csv(run_dir / "fold_metrics.csv", [{key: value for key, value in item.items() if key != "history"} for item in fold_results])
    save_csv(run_dir / "epoch_metrics.csv", all_epoch_rows)
    save_json(outputs_root / "cv_summary.json", summary)
    atomic_json(state_path, {
        "total_folds": total_folds, "completed_folds": completed_folds,
        "active_fold": None, "attempt_count": attempt_count, "status": "complete",
    })
    if static_cache is not None:
        static_cache.cleanup()
        atexit.unregister(static_cache.cleanup)

    logger.info("Saved cross-validation summary to %s", run_dir / "cv_summary.json")

    qualitative_cfg = config.get("qualitative_evaluation", {})
    if bool(qualitative_cfg.get("enabled", False)):
        from src.inference.qualitative_evaluation import run_qualitative_evaluation

        qualitative_result = run_qualitative_evaluation(
            run_dir=run_dir,
            config_path=run_dir / "config.yaml",
            logger=logger,
        )
        if qualitative_result.get("skipped"):
            logger.warning("Skipped qualitative evaluation: %s", qualitative_result.get("reason"))
        else:
            logger.info("Saved qualitative evaluation to %s", qualitative_result["output_dir"])


if __name__ == "__main__":
    main()
