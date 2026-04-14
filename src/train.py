from __future__ import annotations

import argparse
import statistics
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SegmentationPatchDataset, get_train_transforms, get_val_transforms
from src.data.discovery import discover_image_mask_pairs
from src.data.folds import make_grouped_kfold_splits
from src.engine.trainer import Trainer
from src.losses.factory import build_loss
from src.models.factory import build_model
from src.optim.factory import build_optimizer
from src.patching import build_original_image_records, build_patch_records
from src.schedulers.factory import build_scheduler
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_csv, save_json, save_yaml
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fungi segmentation with grouped cross-validation.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_run_dir(runs_root: Path, project_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{project_name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{project_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def log_run_summary(logger, config: dict, device: torch.device, num_images: int) -> None:
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
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
        "Dataset: %s images | folds=%s | patch_size=%s | stride=%s | empty_patch_filter=%s",
        num_images,
        config["cv"]["n_splits"],
        data_cfg["patch_size"],
        data_cfg["stride"],
        data_cfg["filter_empty_patches"],
    )


def log_fold_summary(
    logger,
    fold_index: int,
    total_folds: int,
    train_originals: list,
    val_originals: list,
    train_patch_records: list,
    val_patch_records: list,
) -> None:
    logger.info(
        "Fold %s/%s | train_images=%s | val_images=%s | train_patches=%s | val_patches=%s",
        fold_index + 1,
        total_folds,
        len(train_originals),
        len(val_originals),
        len(train_patch_records),
        len(val_patch_records),
    )
    logger.info(
        "Fold %s split | train_sources=%s | val_sources=%s",
        fold_index + 1,
        ", ".join(record.source_id for record in train_originals),
        ", ".join(record.source_id for record in val_originals),
    )


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
    logger.info("Using device: %s", device)
    logger.info("Run directory: %s", run_dir)

    pairs, diagnostics = discover_image_mask_pairs(
        config["paths"]["images_dir"],
        config["paths"]["masks_dir"],
        config["data"]["image_extensions"],
    )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found.")
    if diagnostics["missing_masks"]:
        logger.warning("Missing masks for %s images.", len(diagnostics["missing_masks"]))
    if diagnostics["missing_images"]:
        logger.warning("Found %s masks without matching images.", len(diagnostics["missing_images"]))

    original_records = build_original_image_records(pairs)
    log_run_summary(logger, config, device, num_images=len(original_records))
    splits = make_grouped_kfold_splits(
        [record.source_id for record in original_records],
        n_splits=int(config["cv"]["n_splits"]),
        shuffle_groups=bool(config["cv"]["shuffle_groups"]),
        random_state=int(config["cv"]["random_state"]),
    )

    fold_results = []
    all_epoch_rows: list[dict[str, float]] = []
    data_cfg = config["data"]
    augmentations_cfg = config.get("augmentations", {})
    total_folds = len(splits)

    for fold_index, (train_sources, val_sources) in enumerate(splits):
        logger.info("Preparing fold %s", fold_index)
        fold_dir = ensure_dir(run_dir / f"fold_{fold_index}")
        tensorboard_dir = ensure_dir(fold_dir / "tensorboard")

        train_originals = [record for record in original_records if record.source_id in set(train_sources)]
        val_originals = [record for record in original_records if record.source_id in set(val_sources)]

        train_patch_records = build_patch_records(
            train_originals,
            patch_size=int(data_cfg["patch_size"]),
            stride=int(data_cfg["stride"]),
            filter_empty_patches=bool(data_cfg["filter_empty_patches"]),
            mask_threshold=int(data_cfg["mask_threshold"]),
            min_foreground_pixels=int(data_cfg["min_foreground_pixels"]),
        )
        val_patch_records = build_patch_records(
            val_originals,
            patch_size=int(data_cfg["patch_size"]),
            stride=int(data_cfg["stride"]),
            filter_empty_patches=False,
            mask_threshold=int(data_cfg["mask_threshold"]),
            min_foreground_pixels=int(data_cfg["min_foreground_pixels"]),
        )
        log_fold_summary(
            logger=logger,
            fold_index=fold_index,
            total_folds=total_folds,
            train_originals=train_originals,
            val_originals=val_originals,
            train_patch_records=train_patch_records,
            val_patch_records=val_patch_records,
        )

        train_dataset = SegmentationPatchDataset(
            records=train_patch_records,
            mask_threshold=int(data_cfg["mask_threshold"]),
            transforms=get_train_transforms(
                data_cfg.get("image_size"),
                augmentations_config=augmentations_cfg,
            ),
        )
        val_dataset = SegmentationPatchDataset(
            records=val_patch_records,
            mask_threshold=int(data_cfg["mask_threshold"]),
            transforms=get_val_transforms(
                data_cfg.get("image_size"),
                augmentations_config=augmentations_cfg,
            ),
        )

        train_loader = make_loader(
            train_dataset,
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            shuffle=True,
        )
        val_loader = make_loader(
            val_dataset,
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            shuffle=False,
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
            logger=logger,
            fold_dir=Path(fold_dir),
            tensorboard_writer=writer,
            fold_index=fold_index,
        )
        fold_result = trainer.fit(train_loader, val_loader, epochs=int(config["train"]["epochs"]))
        writer.close()
        fold_result["fold"] = fold_index
        fold_result["num_train_patches"] = len(train_patch_records)
        fold_result["num_val_patches"] = len(val_patch_records)
        fold_results.append(fold_result)
        all_epoch_rows.extend(
            {"fold": fold_index, **row}
            for row in fold_result["history"]
        )

    dice_values = [float(item["val_dice"]) for item in fold_results]
    iou_values = [float(item["val_iou"]) for item in fold_results]
    summary = {
        "project": project_name,
        "run_dir": str(run_dir),
        "folds": fold_results,
        "mean_dice": statistics.mean(dice_values),
        "std_dice": statistics.pstdev(dice_values) if len(dice_values) > 1 else 0.0,
        "mean_iou": statistics.mean(iou_values),
        "std_iou": statistics.pstdev(iou_values) if len(iou_values) > 1 else 0.0,
        "num_original_images": len(original_records),
    }
    save_json(run_dir / "cv_summary.json", summary)
    save_csv(run_dir / "fold_metrics.csv", [{key: value for key, value in item.items() if key != "history"} for item in fold_results])
    save_csv(run_dir / "epoch_metrics.csv", all_epoch_rows)
    save_json(outputs_root / "cv_summary.json", summary)
    logger.info("Saved cross-validation summary to %s", run_dir / "cv_summary.json")


if __name__ == "__main__":
    main()
