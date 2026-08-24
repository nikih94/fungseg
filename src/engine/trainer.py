from __future__ import annotations

import math
import gc
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import compose_multiclass_mask, get_val_transforms
from src.metrics.segmentation import (
    cldice_score_from_masks,
    dice_score,
    dice_score_from_masks,
    dice_scores,
    iou_score,
    iou_score_from_masks,
    iou_scores,
    join_region_metrics_from_masks,
    multiclass_metrics_from_masks,
    multiclass_predictions,
)
from src.metrics.loss_components import loss_component_metrics
from src.models.wrappers import extract_logits
from src.patching import OriginalImageRecord, _compute_positions, crop_and_pad_array
from src.utils.checkpoint import save_checkpoint
from src.utils.io import save_csv, save_json


def shutdown_dataloader(loader: DataLoader | None) -> None:
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if shutdown_workers is not None:
        with suppress(Exception):
            shutdown_workers()
    with suppress(Exception):
        loader._iterator = None
    gc.collect()


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer,
        scheduler,
        device: torch.device,
        train_config: dict[str, Any],
        loss_config: dict[str, Any] | None,
        logger,
        fold_dir: Path,
        data_config: dict[str, Any],
        augmentations_config: dict[str, Any] | None = None,
        val_original_records: list[OriginalImageRecord] | None = None,
        tensorboard_writer=None,
        fold_index: int = 0,
        segmentation_config: dict[str, Any] | None = None,
        join_masks_config: dict[str, Any] | None = None,
        validation_config: dict[str, Any] | None = None,
        epoch_metrics_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_config = train_config
        self.loss_config = loss_config or {}
        self.logger = logger
        self.fold_dir = fold_dir
        self.data_config = data_config
        self.augmentations_config = augmentations_config or {}
        self.val_original_records = val_original_records or []
        self.tensorboard_writer = tensorboard_writer
        self.fold_index = fold_index
        self.segmentation_config = segmentation_config or {}
        self.join_masks_config = join_masks_config or {}
        self.merge_join_masks = bool(self.join_masks_config.get("merge_with_loci", False))
        if validation_config is None:
            validation_config = {
                "full_image": {
                    "enabled": train_config.get("enable_per_image_validation", True),
                    "interval_epochs": train_config.get("per_image_validation_interval", 1),
                    "monitor": train_config.get("full_image_monitor", {}),
                }
            }
        self.validation_config = validation_config
        self.epoch_metrics_callback = epoch_metrics_callback
        self.segmentation_mode = str(self.segmentation_config.get("mode", "binary")).lower()
        self.class_names = {
            name: int(class_id) for name, class_id in self.segmentation_config.get(
                "classes", {"background": 0, "loci": 1, "inoculum": 2}
            ).items() if name != "background"
        }
        self.monitor = self._normalize_metric_name(train_config.get("monitor", "val_dice_cldice_per_image"))
        self.monitor_mode = train_config.get("monitor_mode", "max")
        interval_checkpoint_config = train_config.get("best_interval_checkpoint", {})
        self.best_interval_checkpoint_enabled = bool(interval_checkpoint_config.get("enabled", False))
        self.best_interval_checkpoint_epochs = max(1, int(interval_checkpoint_config.get("interval_epochs", 10)))
        self.threshold = float(train_config.get("threshold", 0.5))
        self.use_tqdm = bool(train_config.get("use_tqdm", True))
        full_image_config = self.validation_config.get("full_image", {})
        self.enable_per_image_validation = bool(full_image_config.get("enabled", True))
        self.per_image_validation_interval = max(
            1,
            int(full_image_config.get("interval_epochs", 1)),
        )
        full_image_monitor = full_image_config.get("monitor", {})
        if not isinstance(full_image_monitor, dict):
            raise ValueError("validation.full_image.monitor must be a mapping.")
        dice_weight = float(full_image_monitor.get("dice_weight", 0.5))
        cldice_weight = float(full_image_monitor.get("cldice_weight", 0.5))
        if dice_weight < 0.0 or cldice_weight < 0.0 or dice_weight + cldice_weight <= 0.0:
            raise ValueError(
                "validation.full_image.monitor weights must be non-negative and have a positive sum."
            )
        weight_sum = dice_weight + cldice_weight
        self.full_image_dice_weight = dice_weight / weight_sum
        self.full_image_cldice_weight = cldice_weight / weight_sum
        self.scheduler_monitor = self._normalize_metric_name(
            train_config.get("scheduler_monitor", self.monitor)
        )
        monitored_metrics = {self.monitor}
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitored_metrics.add(self.scheduler_monitor)
        if any(metric.endswith("_per_image") for metric in monitored_metrics):
            if not self.enable_per_image_validation:
                raise ValueError(
                    "Full-image monitor metrics require validation.full_image.enabled: true."
                )
            if not self.val_original_records:
                raise ValueError("Full-image monitor metrics require validation image records.")
            if self.per_image_validation_interval != 1:
                raise ValueError(
                    "Full-image monitor metrics require validation.full_image.interval_epochs: 1."
                )
        use_amp = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.val_patch_transforms = get_val_transforms(
            self.data_config.get("image_size"),
            augmentations_config=self.augmentations_config,
        )

    def fit(
        self,
        train_loader: DataLoader | None,
        val_loader: DataLoader,
        epochs: int,
        train_loader_factory=None,
    ) -> dict[str, Any]:
        best_metric = -math.inf if self.monitor_mode == "max" else math.inf
        interval_best_metric = -math.inf if self.monitor_mode == "max" else math.inf
        interval_index = 0
        history: list[dict[str, float]] = []
        epoch_rows: list[dict[str, float]] = []
        checkpoint_manifest: dict[str, dict[str, Any]] = {}

        try:
            for epoch in range(1, epochs + 1):
                current_interval_index = self._interval_index(epoch)
                if current_interval_index != interval_index:
                    interval_index = current_interval_index
                    interval_best_metric = -math.inf if self.monitor_mode == "max" else math.inf

                if train_loader_factory is not None:
                    shutdown_dataloader(train_loader)
                    train_loader = train_loader_factory(epoch)
                if train_loader is None:
                    raise RuntimeError("A train DataLoader or train_loader_factory is required.")

                train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch, epochs=epochs)
                if train_loader_factory is not None:
                    shutdown_dataloader(train_loader)
                val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch, epochs=epochs)
                if self._should_run_per_image_validation(epoch):
                    val_metrics.update(self._evaluate_full_images(epoch=epoch, epochs=epochs))
                epoch_metrics = {
                    "epoch": epoch,
                    **self._current_learning_rates(),
                    **train_metrics,
                    **val_metrics,
                }
                self._validate_finite_epoch_metrics(epoch_metrics, epoch)
                history.append(epoch_metrics)
                epoch_rows.append({"fold": self.fold_index, **epoch_metrics})

                current_metric = float(epoch_metrics[self.monitor])
                self._step_scheduler(epoch_metrics)
                is_best = current_metric > best_metric if self.monitor_mode == "max" else current_metric < best_metric
                if is_best:
                    best_metric = current_metric
                    best_path = self.fold_dir / "best.pt"
                    save_checkpoint(
                        best_path,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        epoch_metrics,
                        self.train_config,
                    )
                    checkpoint_manifest[str(best_path.name)] = self._checkpoint_manifest_row(
                        best_path,
                        epoch,
                        "global_best",
                        epoch_metrics,
                    )

                if self.best_interval_checkpoint_enabled:
                    interval_is_best = (
                        current_metric > interval_best_metric
                        if self.monitor_mode == "max"
                        else current_metric < interval_best_metric
                    )
                    if interval_is_best:
                        interval_best_metric = current_metric
                        interval_path = self.fold_dir / self._interval_checkpoint_name(epoch, epochs)
                        save_checkpoint(
                            interval_path,
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            epoch_metrics,
                            self.train_config,
                        )
                        checkpoint_manifest[str(interval_path.name)] = self._checkpoint_manifest_row(
                            interval_path,
                            epoch,
                            "interval_best",
                            epoch_metrics,
                            total_epochs=epochs,
                        )

                last_path = self.fold_dir / "last.pt"
                save_checkpoint(
                    last_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    epoch_metrics,
                    self.train_config,
                )
                checkpoint_manifest[str(last_path.name)] = self._checkpoint_manifest_row(
                    last_path,
                    epoch,
                    "last",
                    epoch_metrics,
                )

                self._persist_training_progress(
                    best_metric=best_metric,
                    history=history,
                    epoch_rows=epoch_rows,
                    checkpoint_manifest=checkpoint_manifest,
                )
                if self.epoch_metrics_callback is not None:
                    self.epoch_metrics_callback(epoch_metrics)
                self._log_tensorboard(epoch_metrics, epoch)
                self.logger.info(
                    "Epoch %s/%s - %s train_loss=%.4f val_loss=%.4f val_dice_per_patch=%.4f val_iou_per_patch=%.4f val_dice_macro_resolution=%.4f val_dice_per_image=%s val_cldice_loci_per_image=%s val_dice_cldice_per_image=%s val_iou_per_image=%s",
                    epoch,
                    epochs,
                    self._format_learning_rates(epoch_metrics),
                    epoch_metrics["train_loss"],
                    epoch_metrics["val_loss"],
                    epoch_metrics["val_dice_per_patch"],
                    epoch_metrics["val_iou_per_patch"],
                    epoch_metrics["val_dice_macro_resolution"],
                    self._format_optional_metric(epoch_metrics.get("val_dice_per_image")),
                    self._format_optional_metric(epoch_metrics.get("val_cldice_loci_per_image")),
                    self._format_optional_metric(epoch_metrics.get("val_dice_cldice_per_image")),
                    self._format_optional_metric(epoch_metrics.get("val_iou_per_image")),
                )
        finally:
            shutdown_dataloader(train_loader)
            shutdown_dataloader(val_loader)

        self._persist_training_progress(
            best_metric=best_metric,
            history=history,
            epoch_rows=epoch_rows,
            checkpoint_manifest=checkpoint_manifest,
        )
        metrics_payload = {
            "best_metric": best_metric,
            "monitor": self.monitor,
            "history": history,
            "best_epoch": self._best_epoch(history),
        }
        best_epoch = metrics_payload["best_epoch"]
        best_metrics = next(item for item in history if item["epoch"] == best_epoch)
        return {
            "history": history,
            "best_epoch": best_epoch,
            **best_metrics,
        }

    def _persist_training_progress(
        self,
        *,
        best_metric: float,
        history: list[dict[str, float]],
        epoch_rows: list[dict[str, float]],
        checkpoint_manifest: dict[str, dict[str, Any]],
    ) -> None:
        """Refresh fold artifacts so completed epochs are visible during training."""
        metrics_payload = {
            "best_metric": best_metric,
            "monitor": self.monitor,
            "history": history,
            "best_epoch": self._best_epoch(history),
        }
        save_json(self.fold_dir / "metrics.json", metrics_payload)
        save_csv(self.fold_dir / "metrics.csv", epoch_rows)
        manifest_rows = sorted(
            checkpoint_manifest.values(),
            key=lambda row: (
                int(row["epoch_start"]),
                int(row["epoch_end"]),
                str(row["reason"]),
                str(row["checkpoint"]),
            ),
        )
        save_csv(self.fold_dir / "checkpoint_manifest.csv", manifest_rows)
        save_json(
            self.fold_dir / "checkpoint_manifest.json",
            {"checkpoints": manifest_rows},
        )

    def evaluate(
        self,
        loader: DataLoader,
        original_records: list[OriginalImageRecord] | None = None,
        stage: str = "test",
    ) -> dict[str, float]:
        metrics = self._run_epoch(loader, training=False, epoch=1, epochs=1, stage_name=stage)
        if original_records:
            metrics.update(
                self._evaluate_full_images(
                    epoch=1,
                    epochs=1,
                    original_records=original_records,
                    stage=stage,
                )
            )
        shutdown_dataloader(loader)
        return metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
        epochs: int,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_samples = 0
        bucket_dice_totals: dict[str, float] = {}
        bucket_iou_totals: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}
        component_totals: dict[str, float] = {}

        autocast_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cpu"
        stage = stage_name or ("train" if training else "val")
        progress = None
        iterator = loader
        if self.use_tqdm:
            progress = tqdm(
                loader,
                desc=f"Fold {self.fold_index} | Epoch {epoch}/{epochs} | {stage}",
                leave=False,
            )
            iterator = progress

        try:
            for batch_index, batch in enumerate(iterator, start=1):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                geometry_weights = batch.get("loss_weight")
                if geometry_weights is not None:
                    geometry_weights = geometry_weights.to(self.device, non_blocking=True)
                batch_size = int(masks.shape[0])

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                context = torch.enable_grad() if training else torch.no_grad()
                with context:
                    with torch.amp.autocast(device_type=autocast_device, enabled=self.use_amp):
                        logits = extract_logits(self.model(images))
                        loss = (
                            self.loss_fn(logits, masks, geometry_weights=geometry_weights)
                            if geometry_weights is not None
                            else self.loss_fn(logits, masks)
                        )

                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            f"Fold {self.fold_index} epoch {epoch} {stage} batch "
                            f"{batch_index} produced non-finite loss: {loss.item()}."
                        )

                    if training:
                        self.scaler.scale(loss).backward()
                        grad_clip = self.train_config.get("grad_clip")
                        if grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                total_loss += float(loss.item()) * batch_size
                if self.segmentation_mode == "multiclass":
                    batch_task_metrics = multiclass_metrics_from_masks(
                        multiclass_predictions(logits.detach()), masks.detach(), self.class_names
                    )
                    total_dice += batch_task_metrics["dice_macro_foreground"] * batch_size
                    total_iou += batch_task_metrics["iou_macro_foreground"] * batch_size
                    for metric_name, metric_value in batch_task_metrics.items():
                        component_totals[metric_name] = (
                            component_totals.get(metric_name, 0.0) + float(metric_value) * batch_size
                        )
                else:
                    total_dice += dice_score(logits.detach(), masks.detach(), threshold=self.threshold) * batch_size
                    total_iou += iou_score(logits.detach(), masks.detach(), threshold=self.threshold) * batch_size
                with torch.no_grad():
                    batch_components = loss_component_metrics(
                        logits.detach(),
                        masks.detach(),
                        self.loss_config,
                        geometry_weights=(
                            geometry_weights.detach() if geometry_weights is not None else None
                        ),
                    )
                for metric_name, metric_value in batch_components.items():
                    if not math.isfinite(float(metric_value)):
                        raise FloatingPointError(
                            f"Fold {self.fold_index} epoch {epoch} {stage} batch "
                            f"{batch_index} produced non-finite loss component "
                            f"{metric_name}: {metric_value}."
                        )
                    component_totals[metric_name] = (
                        component_totals.get(metric_name, 0.0) + float(metric_value) * batch_size
                    )
                if not training and "resolution_bucket" in batch and self.segmentation_mode != "multiclass":
                    batch_buckets = self._as_string_list(batch["resolution_bucket"])
                    batch_dice_scores = dice_scores(
                        logits.detach(),
                        masks.detach(),
                        threshold=self.threshold,
                    ).detach().cpu().tolist()
                    batch_iou_scores = iou_scores(
                        logits.detach(),
                        masks.detach(),
                        threshold=self.threshold,
                    ).detach().cpu().tolist()
                    for bucket, dice_value, iou_value in zip(batch_buckets, batch_dice_scores, batch_iou_scores):
                        bucket_dice_totals[bucket] = bucket_dice_totals.get(bucket, 0.0) + float(dice_value)
                        bucket_iou_totals[bucket] = bucket_iou_totals.get(bucket, 0.0) + float(iou_value)
                        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                num_samples += batch_size
                if progress is not None:
                    progress.set_postfix(
                        lr=self._progress_learning_rates(),
                        loss=f"{total_loss / num_samples:.4f}",
                        dice=f"{total_dice / num_samples:.4f}",
                        iou=f"{total_iou / num_samples:.4f}",
                    )
                del batch, images, masks, logits, loss
        finally:
            if progress is not None:
                progress.close()
            del iterator
            gc.collect()

        prefix = stage
        divisor = max(num_samples, 1)
        metrics = {
            f"{prefix}_loss": total_loss / divisor,
            f"{prefix}_dice_per_patch": total_dice / divisor,
            f"{prefix}_iou_per_patch": total_iou / divisor,
        }
        metrics.update(
            {
                f"{prefix}_{metric_name}": metric_total / divisor
                for metric_name, metric_total in sorted(component_totals.items())
            }
        )
        if not training and bucket_counts:
            bucket_dice_values = []
            bucket_iou_values = []
            for bucket in sorted(bucket_counts):
                dice_value = bucket_dice_totals[bucket] / bucket_counts[bucket]
                iou_value = bucket_iou_totals[bucket] / bucket_counts[bucket]
                metrics[f"{stage}_dice_{bucket}"] = dice_value
                metrics[f"{stage}_iou_{bucket}"] = iou_value
                bucket_dice_values.append(dice_value)
                bucket_iou_values.append(iou_value)
            metrics[f"{stage}_dice_macro_resolution"] = sum(bucket_dice_values) / len(bucket_dice_values)
            metrics[f"{stage}_iou_macro_resolution"] = sum(bucket_iou_values) / len(bucket_iou_values)
        elif not training:
            metrics[f"{stage}_dice_macro_resolution"] = metrics[f"{stage}_dice_per_patch"]
            metrics[f"{stage}_iou_macro_resolution"] = metrics[f"{stage}_iou_per_patch"]
        if self.segmentation_mode == "multiclass":
            metrics[f"{stage}_dice_macro_foreground"] = metrics.get(
                f"{stage}_dice_macro_foreground", metrics[f"{stage}_dice_per_patch"]
            )
            metrics[f"{stage}_iou_macro_foreground"] = metrics.get(
                f"{stage}_iou_macro_foreground", metrics[f"{stage}_iou_per_patch"]
            )
        return metrics

    def _evaluate_full_images(
        self,
        epoch: int,
        epochs: int,
        original_records: list[OriginalImageRecord] | None = None,
        stage: str = "val",
    ) -> dict[str, float]:
        patch_size = int(self.data_config["patch_size"])
        stride = int(self.data_config["stride"])
        mask_threshold = int(self.data_config["mask_threshold"])
        total_dice = 0.0
        total_iou = 0.0
        total_cldice = 0.0
        total_join_dice = 0.0
        total_join_iou = 0.0
        num_join_images = 0
        num_images = 0
        records = self.val_original_records if original_records is None else original_records

        iterator = records
        if self.use_tqdm:
            iterator = tqdm(
                records,
                desc=f"Fold {self.fold_index} | Epoch {epoch}/{epochs} | {stage}_full_image",
                leave=False,
            )

        self.model.eval()
        with torch.no_grad():
            for record in iterator:
                with Image.open(record.image_path) as image:
                    image_array = np.array(image.convert("RGB"))
                if self.segmentation_mode == "multiclass":
                    if not record.mask_paths:
                        raise ValueError("Multiclass full-image evaluation requires named masks.")
                    with Image.open(record.mask_paths["loci"]) as mask:
                        loci_array = np.array(mask.convert("L"), dtype=np.uint8)
                    with Image.open(record.mask_paths["inoculum"]) as mask:
                        inoculum_array = np.array(mask.convert("L"), dtype=np.uint8)
                    join_array = None
                    if "join" in record.mask_paths:
                        with Image.open(record.mask_paths["join"]) as mask:
                            join_array = np.array(mask.convert("L"), dtype=np.uint8)
                    mask_array, _ = compose_multiclass_mask(
                        loci_array,
                        inoculum_array,
                        mask_threshold,
                        join_mask=join_array,
                        merge_join_masks=getattr(self, "merge_join_masks", False),
                    )
                    height, width = mask_array.shape
                    probability_sum = np.zeros((3, height, width), dtype=np.float32)
                else:
                    with Image.open(record.mask_path) as mask:
                        mask_array = np.array(mask.convert("L"), dtype=np.uint8)
                    height, width = mask_array.shape
                    probability_sum = np.zeros((height, width), dtype=np.float32)
                probability_count = np.zeros((height, width), dtype=np.float32)
                xs = _compute_positions(width, patch_size, stride)
                ys = _compute_positions(height, patch_size, stride)

                for y in ys:
                    for x in xs:
                        image_patch = crop_and_pad_array(image_array, x, y, patch_size)
                        transformed = self.val_patch_transforms(
                            image=image_patch,
                            mask=np.zeros((patch_size, patch_size), dtype=np.float32),
                        )
                        image_tensor = transformed["image"].unsqueeze(0).to(self.device, non_blocking=True)

                        with torch.amp.autocast(
                            device_type=self.device.type if self.device.type in {"cuda", "cpu"} else "cpu",
                            enabled=self.use_amp,
                        ):
                            logits = extract_logits(self.model(image_tensor))
                            probabilities = (
                                torch.softmax(logits, dim=1)
                                if self.segmentation_mode == "multiclass"
                                else torch.sigmoid(logits)
                            )

                        if probabilities.shape[-2:] != (patch_size, patch_size):
                            probabilities = F.interpolate(
                                probabilities,
                                size=(patch_size, patch_size),
                                mode="bilinear",
                                align_corners=False,
                            )

                        probability_patch = probabilities.squeeze(0).cpu().numpy().astype(np.float32)
                        if self.segmentation_mode != "multiclass":
                            probability_patch = probability_patch.squeeze(0)
                        valid_height = min(patch_size, height - y)
                        valid_width = min(patch_size, width - x)
                        if self.segmentation_mode == "multiclass":
                            probability_sum[:, y : y + valid_height, x : x + valid_width] += probability_patch[
                                :, :valid_height, :valid_width
                            ]
                        else:
                            probability_sum[y : y + valid_height, x : x + valid_width] += probability_patch[
                                :valid_height, :valid_width
                            ]
                        probability_count[y : y + valid_height, x : x + valid_width] += 1.0

                if self.segmentation_mode == "multiclass":
                    averaged_probabilities = probability_sum / np.clip(
                        probability_count[None, ...], a_min=1.0, a_max=None
                    )
                    prediction_mask = torch.from_numpy(averaged_probabilities.argmax(axis=0))
                    target_mask = torch.from_numpy(mask_array.astype(np.int64))
                    image_metrics = multiclass_metrics_from_masks(
                        prediction_mask, target_mask, self.class_names
                    )
                    image_dice = image_metrics["dice_macro_foreground"]
                    image_iou = image_metrics["iou_macro_foreground"]
                    image_cldice = image_metrics["cldice_loci"]
                    join_metrics = join_region_metrics_from_masks(
                        prediction_mask,
                        target_mask,
                        None if join_array is None else torch.from_numpy(join_array > mask_threshold),
                        loci_class_id=self.class_names.get("loci", 1),
                    )
                    if join_metrics["dice_join"] is not None:
                        total_join_dice += float(join_metrics["dice_join"])
                        total_join_iou += float(join_metrics["iou_join"])
                        num_join_images += 1
                else:
                    averaged_probabilities = probability_sum / np.clip(
                        probability_count, a_min=1.0, a_max=None
                    )
                    prediction_mask = torch.from_numpy(
                        (averaged_probabilities >= self.threshold).astype(np.float32)
                    )
                    target_mask = torch.from_numpy(
                        (mask_array > mask_threshold).astype(np.float32)
                    )
                    image_dice = dice_score_from_masks(prediction_mask, target_mask)
                    image_iou = iou_score_from_masks(prediction_mask, target_mask)
                    image_cldice = cldice_score_from_masks(prediction_mask, target_mask)
                total_dice += image_dice
                total_iou += image_iou
                total_cldice += image_cldice
                num_images += 1

                if self.use_tqdm:
                    mean_dice = total_dice / max(num_images, 1)
                    mean_cldice = total_cldice / max(num_images, 1)
                    iterator.set_postfix(
                        dice=f"{mean_dice:.4f}",
                        cldice=f"{mean_cldice:.4f}",
                        score=f"{self._combined_full_image_score(mean_dice, mean_cldice):.4f}",
                    )

        divisor = max(num_images, 1)
        mean_dice = total_dice / divisor
        mean_iou = total_iou / divisor
        mean_cldice = total_cldice / divisor
        result = {
            f"{stage}_dice_per_image": mean_dice,
            f"{stage}_iou_per_image": mean_iou,
            f"{stage}_cldice_per_image": mean_cldice,
            f"{stage}_dice_cldice_per_image": self._combined_full_image_score(
                mean_dice, mean_cldice
            ),
        }
        if self.segmentation_mode == "multiclass":
            result[f"{stage}_dice_macro_foreground_per_image"] = mean_dice
            result[f"{stage}_iou_macro_foreground_per_image"] = mean_iou
            result[f"{stage}_cldice_loci_per_image"] = mean_cldice
            result[f"{stage}_join_images"] = num_join_images
            result[f"{stage}_dice_join_per_image"] = (
                total_join_dice / num_join_images if num_join_images else None
            )
            result[f"{stage}_iou_join_per_image"] = (
                total_join_iou / num_join_images if num_join_images else None
            )
        elif str(self.segmentation_config.get("target", "")).lower() == "loci":
            result[f"{stage}_cldice_loci_per_image"] = mean_cldice
        return result

    def _combined_full_image_score(self, dice: float, cldice: float) -> float:
        return (
            self.full_image_dice_weight * float(dice)
            + self.full_image_cldice_weight * float(cldice)
        )

    def _should_run_per_image_validation(self, epoch: int) -> bool:
        return (
            self.enable_per_image_validation
            and bool(self.val_original_records)
            and epoch % self.per_image_validation_interval == 0
        )

    def _step_scheduler(self, epoch_metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(epoch_metrics[self.scheduler_monitor])
            return
        self.scheduler.step()

    @staticmethod
    def _validate_finite_epoch_metrics(
        epoch_metrics: dict[str, float],
        epoch: int,
    ) -> None:
        for metric_name, metric_value in epoch_metrics.items():
            if metric_value is None or not isinstance(metric_value, (int, float)):
                continue
            if not math.isfinite(float(metric_value)):
                raise FloatingPointError(
                    f"Epoch {epoch} produced non-finite metric "
                    f"'{metric_name}': {metric_value}."
                )

    def _best_epoch(self, history: list[dict[str, float]]) -> int:
        reverse = self.monitor_mode == "max"
        ranked = sorted(history, key=lambda item: item[self.monitor], reverse=reverse)
        return int(ranked[0]["epoch"])

    def _log_tensorboard(self, epoch_metrics: dict[str, float], epoch: int) -> None:
        if self.tensorboard_writer is None:
            return
        for key, value in epoch_metrics.items():
            if key == "epoch":
                continue
            if value is None:
                continue
            self.tensorboard_writer.add_scalar(key, value, epoch)

    def _current_learning_rates(self) -> dict[str, float]:
        named_rates = {
            str(group["group_name"]): float(group["lr"])
            for group in self.optimizer.param_groups
            if group.get("group_name") is not None
        }
        if {"encoder", "decoder"} <= named_rates.keys():
            return {
                "lr": named_rates["decoder"],
                "encoder_lr": named_rates["encoder"],
                "decoder_lr": named_rates["decoder"],
            }
        return {"lr": float(self.optimizer.param_groups[0]["lr"])}

    def _progress_learning_rates(self) -> str:
        learning_rates = self._current_learning_rates()
        if "encoder_lr" in learning_rates:
            return (
                f"enc:{learning_rates['encoder_lr']:.2e}/"
                f"dec:{learning_rates['decoder_lr']:.2e}"
            )
        return f"{learning_rates['lr']:.2e}"

    @staticmethod
    def _format_learning_rates(epoch_metrics: dict[str, float]) -> str:
        if "encoder_lr" in epoch_metrics:
            return (
                f"encoder_lr={epoch_metrics['encoder_lr']:.8f} "
                f"decoder_lr={epoch_metrics['decoder_lr']:.8f}"
            )
        return f"lr={epoch_metrics['lr']:.8f}"

    @staticmethod
    def _normalize_metric_name(metric_name: str) -> str:
        legacy_map = {
            "train_dice": "train_dice_per_patch",
            "train_iou": "train_iou_per_patch",
            "val_dice": "val_dice_per_patch",
            "val_iou": "val_iou_per_patch",
        }
        return legacy_map.get(metric_name, metric_name)

    @staticmethod
    def _format_optional_metric(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    @staticmethod
    def _as_string_list(value) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(item) for item in value]

    def _interval_index(self, epoch: int) -> int:
        return (epoch - 1) // self.best_interval_checkpoint_epochs

    def _interval_bounds(self, epoch: int, total_epochs: int | None = None) -> tuple[int, int]:
        interval_start = self._interval_index(epoch) * self.best_interval_checkpoint_epochs + 1
        interval_end = interval_start + self.best_interval_checkpoint_epochs - 1
        if total_epochs is not None:
            interval_end = min(interval_end, total_epochs)
        return interval_start, interval_end

    def _interval_checkpoint_name(self, epoch: int, total_epochs: int | None = None) -> str:
        interval_start, interval_end = self._interval_bounds(epoch, total_epochs=total_epochs)
        return f"best_epochs_{interval_start:03d}_{interval_end:03d}.pt"

    def _checkpoint_manifest_row(
        self,
        path: Path,
        epoch: int,
        reason: str,
        metrics: dict[str, Any],
        total_epochs: int | None = None,
    ) -> dict[str, Any]:
        if reason == "interval_best":
            epoch_start, epoch_end = self._interval_bounds(epoch, total_epochs=total_epochs)
        else:
            epoch_start = epoch
            epoch_end = epoch
        row: dict[str, Any] = {
            "checkpoint": path.name,
            "path": str(path),
            "reason": reason,
            "epoch": epoch,
            "epoch_start": epoch_start,
            "epoch_end": epoch_end,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "monitor_value": metrics.get(self.monitor),
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)) or value is None:
                row[key] = value
        return row
