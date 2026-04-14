from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics.segmentation import dice_score, iou_score
from src.models.wrappers import extract_logits
from src.utils.checkpoint import save_checkpoint
from src.utils.io import save_csv, save_json


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer,
        scheduler,
        device: torch.device,
        train_config: dict[str, Any],
        logger,
        fold_dir: Path,
        tensorboard_writer=None,
        fold_index: int = 0,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_config = train_config
        self.logger = logger
        self.fold_dir = fold_dir
        self.tensorboard_writer = tensorboard_writer
        self.fold_index = fold_index
        self.monitor = train_config.get("monitor", "val_dice")
        self.monitor_mode = train_config.get("monitor_mode", "max")
        self.threshold = float(train_config.get("threshold", 0.5))
        self.use_tqdm = bool(train_config.get("use_tqdm", True))
        use_amp = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict[str, Any]:
        best_metric = -math.inf if self.monitor_mode == "max" else math.inf
        history: list[dict[str, float]] = []
        epoch_rows: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch, epochs=epochs)
            val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch, epochs=epochs)
            epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(epoch_metrics)
            epoch_rows.append({"fold": self.fold_index, **epoch_metrics})

            current_metric = float(epoch_metrics[self.monitor])
            is_best = current_metric > best_metric if self.monitor_mode == "max" else current_metric < best_metric
            if is_best:
                best_metric = current_metric
                save_checkpoint(
                    self.fold_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    epoch_metrics,
                    self.train_config,
                )

            save_checkpoint(
                self.fold_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                epoch_metrics,
                self.train_config,
            )

            self._step_scheduler(epoch_metrics)
            self._log_tensorboard(epoch_metrics, epoch)
            self.logger.info(
                "Epoch %s/%s - train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f",
                epoch,
                epochs,
                epoch_metrics["train_loss"],
                epoch_metrics["val_loss"],
                epoch_metrics["val_dice"],
                epoch_metrics["val_iou"],
            )

        metrics_payload = {
            "best_metric": best_metric,
            "monitor": self.monitor,
            "history": history,
            "best_epoch": self._best_epoch(history),
        }
        save_json(self.fold_dir / "metrics.json", metrics_payload)
        save_csv(self.fold_dir / "metrics.csv", epoch_rows)
        best_epoch = metrics_payload["best_epoch"]
        best_metrics = next(item for item in history if item["epoch"] == best_epoch)
        return {"history": history, "best_epoch": best_epoch, **best_metrics}

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int, epochs: int) -> dict[str, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0

        autocast_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cpu"
        stage = "train" if training else "val"
        iterator = loader
        if self.use_tqdm:
            iterator = tqdm(
                loader,
                desc=f"Fold {self.fold_index} | Epoch {epoch}/{epochs} | {stage}",
                leave=False,
            )

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
            num_batches += 1
            if self.use_tqdm:
                iterator.set_postfix(
                    loss=f"{total_loss / num_batches:.4f}",
                    dice=f"{total_dice / num_batches:.4f}",
                    iou=f"{total_iou / num_batches:.4f}",
                )

        prefix = stage
        divisor = max(num_batches, 1)
        return {
            f"{prefix}_loss": total_loss / divisor,
            f"{prefix}_dice": total_dice / divisor,
            f"{prefix}_iou": total_iou / divisor,
        }

    def _step_scheduler(self, epoch_metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor_key = self.train_config.get("scheduler_monitor", self.monitor)
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
            self.tensorboard_writer.add_scalar(key, value, epoch)
        self.tensorboard_writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
