from __future__ import annotations

import argparse
import statistics
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SegmentationPatchDataset, get_train_transforms, get_val_transforms
from src.data.discovery import discover_image_mask_pairs, discover_image_mask_sets
from src.data.folds import (
    SplitDefinition,
    make_csv_train_val_test_split,
    make_grouped_kfold_splits,
    make_manual_train_val_split,
)
from src.data.sampling import patch_distribution
from src.engine.trainer import Trainer
from src.losses.factory import build_loss
from src.models.factory import build_model
from src.optim.factory import build_optimizer
from src.patching import build_original_image_records, build_patch_records
from src.schedulers.factory import build_scheduler
from src.utils.config import load_config, resolve_mask_dir
from src.utils.io import ensure_dir, save_csv, save_json, save_yaml
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


TORCH_SHARING_STRATEGY = "file_system"


def configure_torch_multiprocessing() -> str:
    torch.multiprocessing.set_sharing_strategy(TORCH_SHARING_STRATEGY)
    return torch.multiprocessing.get_sharing_strategy()


def _worker_init_fn(_: int) -> None:
    torch.multiprocessing.set_sharing_strategy(TORCH_SHARING_STRATEGY)


def _collect_optional_metric(values: list[float | None]) -> tuple[float | None, float | None]:
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return None, None
    mean_value = statistics.mean(valid_values)
    std_value = statistics.pstdev(valid_values) if len(valid_values) > 1 else 0.0
    return mean_value, std_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fungi segmentation with grouped cross-validation.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
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

    raise ValueError(f"Unsupported split mode: {split_mode}. Expected 'csv', 'train_val', or 'kfold'.")


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


def create_run_dir(runs_root: Path, project_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{project_name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{project_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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

    logger.info("Training summary")
    logger.info(
        "Model: %s | encoder=%s | epochs=%s | batch_size=%s | optimizer=%s | lr=%s | device=%s",
        model_cfg["name"],
        model_cfg.get("encoder_name", "-"),
        train_cfg["epochs"],
        data_cfg["batch_size"],
        optimizer_cfg["name"],
        optimizer_cfg["lr"],
        device,
    )
    logger.info(
        "Dataset: %s images | target=%s | masks=%s | split_mode=%s | patch_size=%s | stride=%s | empty_patch_filter=%s",
        num_images,
        config.get("segmentation", {}).get("target", "legacy"),
        mask_dir,
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
    val_patch_records: list,
    test_patch_records: list,
) -> None:
    split_label = "Fold" if split_mode == "kfold" else "Split"
    logger.info(
        "%s %s/%s | train_images=%s | val_images=%s | test_images=%s | train_patches=%s | val_patches=%s | test_patches=%s",
        split_label,
        fold_index + 1,
        total_folds,
        len(train_originals),
        len(val_originals),
        len(test_originals),
        len(train_patch_records),
        len(val_patch_records),
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
    config = load_config(args.config)

    set_seed(int(config["train"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "auto")))

    project_name = config["project"]["name"]
    runs_root = ensure_dir(Path(config["paths"]["runs_dir"]))
    run_dir = create_run_dir(runs_root, project_name)
    outputs_root = ensure_dir(Path(config["paths"]["outputs_dir"]) / project_name)
    logger = setup_logger("train", run_dir / "logs")
    save_yaml(run_dir / "config.yaml", config)
    sharing_strategy = configure_torch_multiprocessing()
    logger.info("Using device: %s", device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Torch multiprocessing sharing strategy: %s", sharing_strategy)

    segmentation_mode = str(config.get("segmentation", {}).get("mode", "binary")).lower()
    if segmentation_mode == "multiclass":
        mask_dir = Path(config["paths"]["mask_dirs"]["loci"])
        pairs, diagnostics = discover_image_mask_sets(
            config["paths"]["images_dir"],
            {
                "loci": config["paths"]["mask_dirs"]["loci"],
                "inoculum": config["paths"]["mask_dirs"]["inoculum"],
            },
            config["data"]["image_extensions"],
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
    else:
        if diagnostics["missing_masks"]:
            logger.warning("Missing masks for %s images.", len(diagnostics["missing_masks"]))
        if diagnostics["missing_images"]:
            logger.warning("Found %s masks without matching images.", len(diagnostics["missing_images"]))

    original_records = build_original_image_records(pairs)
    splits, split_mode = build_splits(config, original_records)
    manifest_rows = split_manifest_rows(splits)
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

    fold_results = []
    all_epoch_rows: list[dict[str, float]] = []
    data_cfg = config["data"]
    patching_cfg = config["patching"]
    augmentations_cfg = config.get("augmentations", {})
    total_folds = len(splits)

    for fold_index, split in enumerate(splits):
        logger.info("Preparing fold %s", fold_index)
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        tensorboard_dir = ensure_dir(fold_dir / "tensorboard")
        train_sources = split.train_sources
        val_sources = split.val_sources
        test_sources = split.test_sources

        train_originals = [record for record in original_records if record.source_id in set(train_sources)]
        val_originals = [record for record in original_records if record.source_id in set(val_sources)]
        test_originals = [record for record in original_records if record.source_id in set(test_sources)]

        train_patch_records = build_patch_records(
            train_originals,
            patching_cfg,
            phase="train",
            epoch=1,
            base_seed=int(config["train"]["seed"]),
        )
        val_patch_records = build_patch_records(
            val_originals,
            patching_cfg,
            phase="validation",
        )
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
            train_patch_records=train_patch_records,
            val_patch_records=val_patch_records,
            test_patch_records=test_patch_records,
        )

        train_dataset = SegmentationPatchDataset(
            records=train_patch_records,
            mask_threshold=int(patching_cfg["mask_threshold"]),
            transforms=get_train_transforms(
                data_cfg.get("image_size"),
                augmentations_config=augmentations_cfg,
            ),
            image_resampling=str(patching_cfg.get("image_resampling", "lanczos")),
            mask_resampling=str(patching_cfg.get("mask_resampling", "foreground_preserving")),
            segmentation_mode=segmentation_mode,
        )
        val_dataset = SegmentationPatchDataset(
            records=val_patch_records,
            mask_threshold=int(patching_cfg["mask_threshold"]),
            transforms=get_val_transforms(
                data_cfg.get("image_size"),
                augmentations_config=augmentations_cfg,
            ),
            image_resampling=str(patching_cfg.get("image_resampling", "lanczos")),
            mask_resampling=str(patching_cfg.get("mask_resampling", "foreground_preserving")),
            segmentation_mode=segmentation_mode,
        )

        patch_diagnostics = {
            "train": patch_distribution(train_patch_records),
            "val": patch_distribution(val_patch_records),
            "test": patch_distribution(test_patch_records),
        }
        save_json(fold_dir / "patch_distribution.json", patch_diagnostics)

        def make_train_loader_for_epoch(epoch: int) -> DataLoader:
            epoch_records = build_patch_records(
                train_originals,
                patching_cfg,
                phase="train",
                epoch=epoch,
                base_seed=int(config["train"]["seed"]),
            )
            train_dataset.set_records(epoch_records)
            return make_loader(
                train_dataset,
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

        val_loader = make_loader(
            val_dataset,
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            shuffle=False,
            persistent_workers=bool(data_cfg.get("persistent_workers", False)),
            prefetch_factor=(
                int(data_cfg["prefetch_factor"])
                if data_cfg.get("prefetch_factor") is not None
                else None
            ),
        )

        model = build_model(config["model"]).to(device)
        loss_fn = build_loss(config["loss"])
        optimizer = build_optimizer(model.parameters(), config["optimizer"])
        scheduler = build_scheduler(optimizer, config["scheduler"])
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_config={
                **config["train"],
                "scheduler_monitor": config["scheduler"].get("monitor", config["train"]["monitor"]),
            },
            loss_config=config["loss"],
            logger=logger,
            fold_dir=Path(fold_dir),
            data_config={**data_cfg, **patching_cfg},
            augmentations_config=augmentations_cfg,
            val_original_records=val_originals,
            tensorboard_writer=writer,
            fold_index=fold_index,
            segmentation_config=config.get("segmentation", {}),
        )
        fold_result = trainer.fit(
            None,
            val_loader,
            epochs=int(config["train"]["epochs"]),
            train_loader_factory=make_train_loader_for_epoch,
        )
        writer.close()
        if test_originals and bool(config.get("test_evaluation", {}).get("enabled", True)):
            from src.test_evaluation import run_test_evaluation

            best_checkpoint_path = fold_dir / "best.pt"
            evaluation_dir = run_dir / "test-evaluation"
            if total_folds > 1:
                evaluation_dir = evaluation_dir / f"fold_{fold_index}"
            test_result = run_test_evaluation(
                best_checkpoint_path,
                config,
                evaluation_dir,
                device,
            )
            fold_result.update({
                "test_dice_per_image": test_result["mean_dice"],
                "test_iou_per_image": test_result["mean_iou"],
            })
            logger.info(
                "Fold %s test evaluation - test_dice_per_image=%.4f test_iou_per_image=%.4f output=%s",
                fold_index,
                test_result["mean_dice"],
                test_result["mean_iou"],
                evaluation_dir,
            )
        fold_result["fold"] = fold_index
        fold_result["num_train_patches"] = len(train_patch_records)
        fold_result["num_val_patches"] = len(val_patch_records)
        fold_result["num_test_patches"] = len(test_patch_records)
        fold_result["num_train_normal_patches"] = sum(
            1 for record in train_patch_records if record.scale_label == "normal"
        )
        fold_result["num_train_scaled_context_patches"] = sum(
            1 for record in train_patch_records if record.scale_label == "scaled_context"
        )
        fold_result["num_val_normal_patches"] = sum(
            1 for record in val_patch_records if record.scale_label == "normal"
        )
        fold_result["num_test_normal_patches"] = sum(
            1 for record in test_patch_records if record.scale_label == "normal"
        )
        fold_results.append(fold_result)
        all_epoch_rows.extend(
            {"fold": fold_index, **row}
            for row in fold_result["history"]
        )

    val_dice_per_patch_values = [float(item["val_dice_per_patch"]) for item in fold_results]
    val_iou_per_patch_values = [float(item["val_iou_per_patch"]) for item in fold_results]
    val_dice_macro_resolution_values = [
        float(item["val_dice_macro_resolution"]) for item in fold_results if item.get("val_dice_macro_resolution") is not None
    ]
    val_iou_macro_resolution_values = [
        float(item["val_iou_macro_resolution"]) for item in fold_results if item.get("val_iou_macro_resolution") is not None
    ]
    val_dice_per_image_mean, val_dice_per_image_std = _collect_optional_metric(
        [item.get("val_dice_per_image") for item in fold_results]
    )
    val_iou_per_image_mean, val_iou_per_image_std = _collect_optional_metric(
        [item.get("val_iou_per_image") for item in fold_results]
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
        "segmentation_target": config.get("segmentation", {}).get("target", "legacy"),
        "mask_dir": str(mask_dir),
        "folds": fold_results,
        "mean_dice_per_patch": statistics.mean(val_dice_per_patch_values),
        "std_dice_per_patch": (
            statistics.pstdev(val_dice_per_patch_values) if len(val_dice_per_patch_values) > 1 else 0.0
        ),
        "mean_iou_per_patch": statistics.mean(val_iou_per_patch_values),
        "std_iou_per_patch": (
            statistics.pstdev(val_iou_per_patch_values) if len(val_iou_per_patch_values) > 1 else 0.0
        ),
        "mean_dice_macro_resolution": (
            statistics.mean(val_dice_macro_resolution_values) if val_dice_macro_resolution_values else None
        ),
        "std_dice_macro_resolution": (
            statistics.pstdev(val_dice_macro_resolution_values)
            if len(val_dice_macro_resolution_values) > 1
            else 0.0
        ),
        "mean_iou_macro_resolution": (
            statistics.mean(val_iou_macro_resolution_values) if val_iou_macro_resolution_values else None
        ),
        "std_iou_macro_resolution": (
            statistics.pstdev(val_iou_macro_resolution_values)
            if len(val_iou_macro_resolution_values) > 1
            else 0.0
        ),
        "mean_dice_per_image": val_dice_per_image_mean,
        "std_dice_per_image": val_dice_per_image_std,
        "mean_iou_per_image": val_iou_per_image_mean,
        "std_iou_per_image": val_iou_per_image_std,
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
    logger.info("Saved cross-validation summary to %s", run_dir / "cv_summary.json")

    qualitative_cfg = config.get("qualitative_evaluation", {})
    if bool(qualitative_cfg.get("enabled", False)):
        from src.qualitative_evaluation import run_qualitative_evaluation

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
