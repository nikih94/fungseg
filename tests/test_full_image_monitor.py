from __future__ import annotations

import math
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.engine.trainer import Trainer
from src.metrics.segmentation import multiclass_metrics_from_masks
from src.patching import OriginalImageRecord, build_patch_records
from src.schedulers.factory import build_scheduler
from src.train import (
    build_fast_validation_patching_config,
    select_full_image_validation_records,
)
from src.utils.config import load_config


class _SilentLogger:
    def info(self, *args, **kwargs) -> None:
        return None


class _StubTrainer(Trainer):
    def __init__(
        self,
        fold_dir: Path,
        scores: list[float],
        epoch_metrics_callback=None,
    ) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = None
        self.scheduler_monitor = "val_dice_cldice_per_image"
        self.monitor = "val_dice_cldice_per_image"
        self.monitor_mode = "max"
        self.best_interval_checkpoint_enabled = True
        self.best_interval_checkpoint_epochs = 10
        self.fold_dir = fold_dir
        self.train_config = {}
        self.tensorboard_writer = None
        self.logger = _SilentLogger()
        self.fold_index = 0
        self.scores = scores
        self.epoch_metrics_callback = epoch_metrics_callback

    def _run_epoch(
        self,
        loader,
        training: bool,
        epoch: int,
        epochs: int,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        if training:
            return {
                "train_loss": float(epoch),
                "train_dice_per_patch": 0.1,
                "train_iou_per_patch": 0.1,
            }
        return {
            "val_loss": float(epoch),
            "val_dice_per_patch": 0.8 - (0.1 * epoch),
            "val_iou_per_patch": 0.7 - (0.1 * epoch),
            "val_dice_macro_resolution": 0.8 - (0.1 * epoch),
            "val_iou_macro_resolution": 0.7 - (0.1 * epoch),
        }

    def _evaluate_full_images(
        self,
        epoch: int,
        epochs: int,
        original_records=None,
        stage: str = "val",
    ) -> dict[str, float]:
        score = self.scores[epoch - 1]
        return {
            "val_dice_per_image": score,
            "val_iou_per_image": score,
            "val_cldice_per_image": score,
            "val_cldice_loci_per_image": score,
            "val_dice_cldice_per_image": score,
        }

    def _should_run_per_image_validation(self, epoch: int) -> bool:
        return True

    def _log_tensorboard(self, epoch_metrics: dict[str, float], epoch: int) -> None:
        return None


class FullImageMonitorTests(unittest.TestCase):
    @staticmethod
    def _raw_transform(*, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()
        return {"image": image_tensor, "mask": torch.from_numpy(mask)}

    def test_multiclass_full_image_monitor_uses_stitched_argmax_dice_and_cldice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = np.zeros((8, 8), dtype=np.uint8)
            target[3, 1:7] = 1
            target[:2, 6:] = 2
            prediction = target.copy()
            prediction[3, 4] = 0

            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., 0] = 255
            for class_id in (1, 2):
                class_pixels = prediction == class_id
                image[class_pixels, 0] = 0
                image[class_pixels, class_id] = 255
            loci = np.where(target == 1, 255, 0).astype(np.uint8)
            inoculum = np.where(target == 2, 255, 0).astype(np.uint8)
            image_path = root / "image.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(loci).save(loci_path)
            Image.fromarray(inoculum).save(inoculum_path)
            record = OriginalImageRecord(
                "image.png",
                image_path,
                loci_path,
                8,
                8,
                {"loci": loci_path, "inoculum": inoculum_path},
            )

            trainer = Trainer.__new__(Trainer)
            trainer.model = torch.nn.Identity()
            trainer.device = torch.device("cpu")
            trainer.data_config = {"patch_size": 4, "stride": 2, "mask_threshold": 127}
            trainer.val_patch_transforms = self._raw_transform
            trainer.val_original_records = [record]
            trainer.segmentation_mode = "multiclass"
            trainer.segmentation_config = {
                "mode": "multiclass",
                "classes": {"background": 0, "loci": 1, "inoculum": 2},
            }
            trainer.class_names = {"loci": 1, "inoculum": 2}
            trainer.threshold = 0.5
            trainer.use_amp = False
            trainer.use_tqdm = False
            trainer.fold_index = 0
            trainer.full_image_dice_weight = 0.25
            trainer.full_image_cldice_weight = 0.75

            metrics = trainer._evaluate_full_images(epoch=1, epochs=1)

        expected = multiclass_metrics_from_masks(
            torch.from_numpy(prediction.astype(np.int64)),
            torch.from_numpy(target.astype(np.int64)),
            {"loci": 1, "inoculum": 2},
        )
        expected_combined = (
            0.25 * expected["dice_macro_foreground"]
            + 0.75 * expected["cldice_loci"]
        )
        self.assertAlmostEqual(
            metrics["val_dice_macro_foreground_per_image"],
            expected["dice_macro_foreground"],
        )
        self.assertAlmostEqual(
            metrics["val_cldice_loci_per_image"], expected["cldice_loci"]
        )
        self.assertAlmostEqual(
            metrics["val_dice_cldice_per_image"], expected_combined
        )

    def test_reduce_on_plateau_consumes_combined_full_image_score(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_scheduler(
            optimizer,
            {
                "name": "reduce_on_plateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 7,
                "min_lr": 1e-6,
                "monitor": "val_dice_cldice_per_image",
            },
        )
        trainer = Trainer.__new__(Trainer)
        trainer.scheduler = scheduler
        trainer.scheduler_monitor = "val_dice_cldice_per_image"

        trainer._step_scheduler(
            {
                "val_dice_per_patch": 0.99,
                "val_dice_cldice_per_image": 0.42,
            }
        )

        self.assertEqual(scheduler.best, 0.42)

    def test_full_image_monitor_requires_every_epoch_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "interval_epochs: 1"):
            Trainer(
                model=torch.nn.Identity(),
                loss_fn=None,
                optimizer=None,
                scheduler=None,
                device=torch.device("cpu"),
                train_config={
                    "monitor": "val_dice_cldice_per_image",
                    "enable_per_image_validation": True,
                    "per_image_validation_interval": 10,
                },
                loss_config=None,
                logger=None,
                fold_dir=Path("unused"),
                data_config={},
                val_original_records=[object()],
            )

    def test_best_checkpoint_selection_uses_combined_full_image_score(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.monitor = "val_dice_cldice_per_image"
        trainer.monitor_mode = "max"
        history = [
            {
                "epoch": 1,
                "val_dice_per_patch": 0.95,
                "val_dice_cldice_per_image": 0.40,
            },
            {
                "epoch": 2,
                "val_dice_per_patch": 0.70,
                "val_dice_cldice_per_image": 0.80,
            },
        ]

        self.assertEqual(trainer._best_epoch(history), 2)

    def test_smallest_full_image_validation_selection_is_area_sorted_and_deterministic(self) -> None:
        records = [
            OriginalImageRecord("large.png", Path("large.png"), Path("large-mask.png"), 20, 20),
            OriginalImageRecord("b.png", Path("b.png"), Path("b-mask.png"), 10, 10),
            OriginalImageRecord("a.png", Path("a.png"), Path("a-mask.png"), 5, 20),
            OriginalImageRecord("tiny.png", Path("tiny.png"), Path("tiny-mask.png"), 3, 4),
        ]

        selected = select_full_image_validation_records(
            records,
            {"full_image": {"selection": "smallest_area", "max_images": 3}},
        )

        self.assertEqual([record.source_id for record in selected], ["tiny.png", "a.png", "b.png"])
        self.assertEqual(
            select_full_image_validation_records(
                records, {"full_image": {"selection": "all", "max_images": None}}
            ),
            records,
        )

    def test_fast_validation_uses_shared_foreground_threshold_and_zero_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
            loci = np.zeros((8, 8), dtype=np.uint8)
            loci[1, 1:3] = 200
            Image.fromarray(loci).save(loci_path)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(inoculum_path)
            record = OriginalImageRecord(
                "image.png",
                image_path,
                loci_path,
                8,
                8,
                {"loci": loci_path, "inoculum": inoculum_path},
            )
            shared = {
                "patch_size": 4,
                "overlap": 2,
                "stride": 2,
                "filter_empty_patches": False,
                "mask_threshold": 127,
                "min_foreground_pixels": 2,
            }
            effective = build_fast_validation_patching_config(
                shared,
                {"fast": {"foreground_only": True, "overlap": 0}},
            )
            records = build_patch_records([record], effective, phase="validation")

        self.assertEqual(effective["stride"], 4)
        self.assertEqual(effective["mask_threshold"], 127)
        self.assertEqual(effective["min_foreground_pixels"], 2)
        self.assertEqual([(record.x, record.y) for record in records], [(0, 0)])

    def test_ready_configs_keep_monitor_dependencies_internally_consistent(self) -> None:
        for filename in (
            "config.yaml",
            "config_segformer_mit_b3.yaml",
            "multiclass-config.yaml",
            "multiclass-segformer-config.yaml",
            "multiclass-segformer-mit-b1-refinement-config.yaml",
            "multiclass-segformer-mit-b2-refinement-config.yaml",
            "multiclass-segformer-mit-b3-geometry-config.yaml",
        ):
            with self.subTest(config=filename):
                config = load_config(filename)
                train_monitor = str(config["train"]["monitor"])
                scheduler_monitor = str(config["scheduler"]["monitor"])
                self.assertEqual(scheduler_monitor, train_monitor)
                if train_monitor.endswith("_per_image"):
                    self.assertTrue(config["validation"]["full_image"]["enabled"])
                    self.assertEqual(
                        config["validation"]["full_image"]["interval_epochs"],
                        1,
                    )
                if train_monitor == "val_dice_cldice_per_image":
                    weights = config["validation"]["full_image"]["monitor"]
                    self.assertGreaterEqual(float(weights["dice_weight"]), 0.0)
                    self.assertGreaterEqual(float(weights["cldice_weight"]), 0.0)
                    self.assertGreater(
                        float(weights["dice_weight"]) + float(weights["cldice_weight"]),
                        0.0,
                    )
                if filename.startswith("multiclass"):
                    self.assertEqual(train_monitor, "val_dice_per_patch")
                    self.assertTrue(
                        config["validation"]["fast"]["foreground_only"]
                    )
                    self.assertEqual(config["validation"]["fast"]["overlap"], 0)
                    self.assertFalse(
                        config["validation"]["full_image"]["enabled"]
                    )


class TrainerLifecycleRegressionTests(unittest.TestCase):
    def test_epoch_artifacts_and_callback_are_refreshed_during_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = Path(tmpdir)
            observed: list[tuple[int, int, int]] = []

            def observe(epoch_metrics: dict[str, float]) -> None:
                with (fold_dir / "metrics.csv").open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    metric_rows = list(csv.DictReader(handle))
                with (fold_dir / "checkpoint_manifest.csv").open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    manifest_rows = list(csv.DictReader(handle))
                observed.append(
                    (
                        int(epoch_metrics["epoch"]),
                        len(metric_rows),
                        len(manifest_rows),
                    )
                )

            trainer = _StubTrainer(
                fold_dir,
                scores=[0.9, 0.2],
                epoch_metrics_callback=observe,
            )
            trainer.fit([object()], [object()], epochs=2)

        self.assertEqual([item[:2] for item in observed], [(1, 1), (2, 2)])
        self.assertTrue(all(manifest_count >= 2 for _, _, manifest_count in observed))

    def test_fit_returns_metrics_from_one_consistent_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _StubTrainer(Path(tmpdir), scores=[0.9, 0.2])

            result = trainer.fit([object()], [object()], epochs=2)

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["val_loss"], 1.0)
        self.assertEqual(result["val_dice_per_image"], 0.9)
        self.assertEqual(result["val_dice_cldice_per_image"], 0.9)

    def test_last_checkpoint_contains_post_scheduler_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _StubTrainer(Path(tmpdir), scores=[1.0, 0.5])
            trainer.scheduler = build_scheduler(
                trainer.optimizer,
                {
                    "name": "reduce_on_plateau",
                    "mode": "max",
                    "factor": 0.5,
                    "patience": 0,
                    "min_lr": 0.001,
                },
            )

            trainer.fit([object()], [object()], epochs=2)
            payload = torch.load(Path(tmpdir) / "last.pt", map_location="cpu")

        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.05)
        self.assertEqual(
            payload["optimizer_state_dict"]["param_groups"][0]["lr"],
            0.05,
        )

    def test_fit_rejects_non_finite_metrics_before_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = Path(tmpdir)
            trainer = _StubTrainer(fold_dir, scores=[math.nan])

            with self.assertRaisesRegex(
                FloatingPointError,
                "non-finite metric 'val_dice_per_image'",
            ):
                trainer.fit([object()], [object()], epochs=1)

            self.assertFalse((fold_dir / "best.pt").exists())
            self.assertFalse((fold_dir / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
