from __future__ import annotations

import math
import gc
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import get_val_transforms
from src.metrics.segmentation import (
    dice_score,
    dice_score_from_masks,
    dice_scores,
    iou_score,
    iou_score_from_masks,
    iou_scores,
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
        self.monitor = self._normalize_metric_name(train_config.get("monitor", "val_dice_per_patch"))
        self.monitor_mode = train_config.get("monitor_mode", "max")
        interval_checkpoint_config = train_config.get("best_interval_checkpoint", {})
        self.best_interval_checkpoint_enabled = bool(interval_checkpoint_config.get("enabled", False))
        self.best_interval_checkpoint_epochs = max(1, int(interval_checkpoint_config.get("interval_epochs", 10)))
        self.threshold = float(train_config.get("threshold", 0.5))
        self.use_tqdm = bool(train_config.get("use_tqdm", True))
        self.enable_per_image_validation = bool(train_config.get("enable_per_image_validation", True))
        self.per_image_validation_interval = max(1, int(train_config.get("per_image_validation_interval", 1)))
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
                    "lr": self._current_lr(),
                    **train_metrics,
                    **val_metrics,
                }
                history.append(epoch_metrics)
                epoch_rows.append({"fold": self.fold_index, **epoch_metrics})

                current_metric = float(epoch_metrics[self.monitor])
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

                self._step_scheduler(epoch_metrics)
                self._log_tensorboard(epoch_metrics, epoch)
                self.logger.info(
                    "Epoch %s/%s - lr=%.8f train_loss=%.4f val_loss=%.4f val_dice_per_patch=%.4f val_iou_per_patch=%.4f val_dice_macro_resolution=%.4f val_dice_per_image=%s val_iou_per_image=%s",
                    epoch,
                    epochs,
                    epoch_metrics["lr"],
                    epoch_metrics["train_loss"],
                    epoch_metrics["val_loss"],
                    epoch_metrics["val_dice_per_patch"],
                    epoch_metrics["val_iou_per_patch"],
                    epoch_metrics["val_dice_macro_resolution"],
                    self._format_optional_metric(epoch_metrics.get("val_dice_per_image")),
                    self._format_optional_metric(epoch_metrics.get("val_iou_per_image")),
                )
        finally:
            shutdown_dataloader(train_loader)
            shutdown_dataloader(val_loader)

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
        save_json(self.fold_dir / "checkpoint_manifest.json", {"checkpoints": manifest_rows})
        best_epoch = metrics_payload["best_epoch"]
        best_metrics = next(item for item in history if item["epoch"] == best_epoch)
        return {
            "history": history,
            "best_epoch": best_epoch,
            **best_metrics,
            "val_dice_per_image": self._latest_metric(history, "val_dice_per_image"),
            "val_iou_per_image": self._latest_metric(history, "val_iou_per_image"),
        }

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int, epochs: int) -> dict[str, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        bucket_dice_totals: dict[str, float] = {}
        bucket_iou_totals: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}
        component_totals: dict[str, float] = {}

        autocast_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cpu"
        stage = "train" if training else "val"
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
            for batch in iterator:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                context = torch.enable_grad() if training else torch.no_grad()
                with context:
                    with torch.amp.autocast(device_type=autocast_device, enabled=self.use_amp):
                        logits = extract_logits(self.model(images))
                        loss = self.loss_fn(logits, masks)

                    if training:
                        self.scaler.scale(loss).backward()
                        grad_clip = self.train_config.get("grad_clip")
                        if grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                total_loss += float(loss.item())
                total_dice += dice_score(logits.detach(), masks.detach(), threshold=self.threshold)
                total_iou += iou_score(logits.detach(), masks.detach(), threshold=self.threshold)
                with torch.no_grad():
                    batch_components = loss_component_metrics(logits.detach(), masks.detach(), self.loss_config)
                for metric_name, metric_value in batch_components.items():
                    component_totals[metric_name] = component_totals.get(metric_name, 0.0) + float(metric_value)
                if not training and "resolution_bucket" in batch:
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
                num_batches += 1
                if progress is not None:
                    progress.set_postfix(
                        lr=f"{self._current_lr():.2e}",
                        loss=f"{total_loss / num_batches:.4f}",
                        dice=f"{total_dice / num_batches:.4f}",
                        iou=f"{total_iou / num_batches:.4f}",
                    )
                del batch, images, masks, logits, loss
        finally:
            if progress is not None:
                progress.close()
            del iterator
            gc.collect()

        prefix = stage
        divisor = max(num_batches, 1)
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
                metrics[f"val_dice_{bucket}"] = dice_value
                metrics[f"val_iou_{bucket}"] = iou_value
                bucket_dice_values.append(dice_value)
                bucket_iou_values.append(iou_value)
            metrics["val_dice_macro_resolution"] = sum(bucket_dice_values) / len(bucket_dice_values)
            metrics["val_iou_macro_resolution"] = sum(bucket_iou_values) / len(bucket_iou_values)
        elif not training:
            metrics["val_dice_macro_resolution"] = metrics["val_dice_per_patch"]
            metrics["val_iou_macro_resolution"] = metrics["val_iou_per_patch"]
        return metrics

    def _evaluate_full_images(self, epoch: int, epochs: int) -> dict[str, float]:
        patch_size = int(self.data_config["patch_size"])
        stride = int(self.data_config["stride"])
        mask_threshold = int(self.data_config["mask_threshold"])
        total_dice = 0.0
        total_iou = 0.0
        num_images = 0

        iterator = self.val_original_records
        if self.use_tqdm:
            iterator = tqdm(
                self.val_original_records,
                desc=f"Fold {self.fold_index} | Epoch {epoch}/{epochs} | val_full_image",
                leave=False,
            )

        self.model.eval()
        with torch.no_grad():
            for record in iterator:
                with Image.open(record.image_path) as image:
                    image_array = np.array(image.convert("RGB"))
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
                            probabilities = torch.sigmoid(logits)

                        if probabilities.shape[-2:] != (patch_size, patch_size):
                            probabilities = F.interpolate(
                                probabilities,
                                size=(patch_size, patch_size),
                                mode="bilinear",
                                align_corners=False,
                            )

                        probability_patch = probabilities.squeeze().cpu().numpy().astype(np.float32)
                        valid_height = min(patch_size, height - y)
                        valid_width = min(patch_size, width - x)
                        probability_sum[y : y + valid_height, x : x + valid_width] += probability_patch[
                            :valid_height, :valid_width
                        ]
                        probability_count[y : y + valid_height, x : x + valid_width] += 1.0

                averaged_probabilities = probability_sum / np.clip(probability_count, a_min=1.0, a_max=None)
                prediction_mask = torch.from_numpy((averaged_probabilities >= self.threshold).astype(np.float32))
                target_mask = torch.from_numpy((mask_array > mask_threshold).astype(np.float32))

                total_dice += dice_score_from_masks(prediction_mask, target_mask)
                total_iou += iou_score_from_masks(prediction_mask, target_mask)
                num_images += 1

                if self.use_tqdm:
                    iterator.set_postfix(
                        dice=f"{total_dice / max(num_images, 1):.4f}",
                        iou=f"{total_iou / max(num_images, 1):.4f}",
                    )

        divisor = max(num_images, 1)
        return {
            "val_dice_per_image": total_dice / divisor,
            "val_iou_per_image": total_iou / divisor,
        }

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
            monitor_key = self._normalize_metric_name(self.train_config.get("scheduler_monitor", self.monitor))
            self.scheduler.step(epoch_metrics[monitor_key])
            return
        self.scheduler.step()

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

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

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
    def _latest_metric(history: list[dict[str, float]], key: str) -> float | None:
        for item in reversed(history):
            value = item.get(key)
            if value is not None:
                return float(value)
        return None

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
