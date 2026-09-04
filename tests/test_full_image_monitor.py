from __future__ import annotations

import math
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import src.metrics.segmentation as segmentation_metrics
from src.data.patch_cache import build_static_validation_patch_cache
from src.engine.trainer import Trainer
from src.metrics.segmentation import multiclass_metrics_from_masks
from src.patching import OriginalImageRecord
from src.schedulers.factory import build_scheduler
from src.train import (
    build_full_image_validation_patching_config,
    select_full_image_validation_records,
)
from src.utils.config import load_config


class _SilentLogger:
    def info(self, *args, **kwargs) -> None:
        return None


class _BatchRecordingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(inputs.shape[0]))
        return inputs


class _BatchRecordingCrossEntropy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.iteration_values: list[list[int] | None] = []
        self.geometry_weight_shapes: list[tuple[int, ...] | None] = []

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        geometry_weights: torch.Tensor | None = None,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.batch_sizes.append(int(targets.shape[0]))
        self.iteration_values.append(
            None
            if soft_cldice_iterations is None
            else soft_cldice_iterations.detach().cpu().tolist()
        )
        self.geometry_weight_shapes.append(
            None if geometry_weights is None else tuple(geometry_weights.shape)
        )
        return torch.nn.functional.cross_entropy(logits, targets)


class _BatchRecordingFirstChannel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(inputs.shape[0]))
        return inputs[:, :1]


class _BatchRecordingBCE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.iteration_values: list[list[int] | None] = []
        self.target_foreground_counts: list[list[int]] = []

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_cldice_iterations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.batch_sizes.append(int(targets.shape[0]))
        self.target_foreground_counts.append(
            targets.reshape(targets.shape[0], -1).sum(dim=1).int().tolist()
        )
        self.iteration_values.append(
            None
            if soft_cldice_iterations is None
            else soft_cldice_iterations.detach().cpu().tolist()
        )
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


class _StubTrainer(Trainer):
    def __init__(
        self,
        fold_dir: Path,
        scores: list[float],
        epoch_metrics_callback=None,
        validation_metrics: list[dict[str, float]] | None = None,
    ) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = None
        self.scheduler_monitor = "val_loss"
        self.monitor = "val_dice_cldice_per_image"
        self.monitor_mode = "max"
        self.best_interval_checkpoint_enabled = True
        self.best_interval_checkpoint_epochs = 10
        self.save_last_checkpoint = True
        self.fold_dir = fold_dir
        self.train_config = {}
        self.tensorboard_writer = None
        self.logger = _SilentLogger()
        self.fold_index = 0
        self.scores = scores
        self.validation_metrics = validation_metrics
        self.validation_start_epoch = 1
        self.segmentation_mode = "multiclass"
        self.checkpoint_specs = [
            ("best_current.pt", "val_dice_cldice_per_image", "max", "global_best"),
            ("best_dice.pt", "val_dice_per_image", "max", "dice"),
            ("best_low_cldice.pt", "val_dice_low_cldice_per_image", "max", "low_cldice"),
            ("best_inoculum_compensated.pt", "val_inoculum_compensated_per_image", "max", "inoculum_compensated"),
        ]
        self.validation_epochs: list[int] = []
        self.epoch_metrics_callback = epoch_metrics_callback

    def _run_epoch(
        self,
        loader,
        training: bool,
        epoch: int,
        epochs: int,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        assert training
        return {
            "train_loss": float(epoch),
            "train_dice_per_patch": 0.1,
            "train_iou_per_patch": 0.1,
        }

    def _evaluate_full_images(
        self,
        epoch: int,
        epochs: int,
        original_records=None,
        stage: str = "val",
    ) -> dict[str, float]:
        score = self.scores[epoch - 1]
        self.validation_epochs.append(epoch)
        metrics = {
            "val_loss": 1.0 - score,
            "val_dice_per_image": score,
            "val_iou_per_image": score,
            "val_cldice_per_image": score,
            "val_cldice_loci_per_image": score,
            "val_dice_loci_per_image": score,
            "val_dice_inoculum_per_image": score,
            "val_dice_cldice_per_image": score,
            "val_dice_low_cldice_per_image": score,
            "val_inoculum_compensated_per_image": score,
        }
        if self.validation_metrics is not None:
            metrics.update(self.validation_metrics[epoch - 1])
        return metrics

    def _should_run_per_image_validation(self, epoch: int) -> bool:
        return epoch >= self.validation_start_epoch

    def _log_tensorboard(self, epoch_metrics: dict[str, float], epoch: int) -> None:
        return None


class FullImageMonitorTests(unittest.TestCase):
    @staticmethod
    def _raw_transform(
        *, image: np.ndarray, mask: np.ndarray | None = None
    ) -> dict[str, torch.Tensor]:
        transformed = {
            "image": torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            ).float()
        }
        if mask is not None:
            transformed["mask"] = torch.from_numpy(mask)
        return transformed

    def test_multiclass_full_image_validation_uses_stitched_argmax_and_hard_cldice(self) -> None:
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
            batched_model = _BatchRecordingIdentity()
            batched_loss = _BatchRecordingCrossEntropy()
            trainer.model = batched_model
            trainer.loss_fn = batched_loss
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
            trainer.full_image_dice_weight = 0.7
            trainer.full_image_cldice_weight = 0.3
            trainer.full_image_batch_size = 8
            trainer.target_weight_builder = lambda target: torch.ones_like(
                target, dtype=torch.float32
            )
            trainer.soft_cldice_iterations = {"image.png": 7}
            trainer.default_soft_cldice_iterations = 3

            with patch.object(
                segmentation_metrics,
                "skeletonize",
                wraps=segmentation_metrics.skeletonize,
            ) as skeletonize_mock:
                metrics = trainer._evaluate_full_images(epoch=1, epochs=1)
                single_patch_model = _BatchRecordingIdentity()
                single_patch_loss = _BatchRecordingCrossEntropy()
                trainer.model = single_patch_model
                trainer.loss_fn = single_patch_loss
                trainer.full_image_batch_size = 1
                single_patch_metrics = trainer._evaluate_full_images(epoch=1, epochs=1)

                validation_cache = build_static_validation_patch_cache(
                    [record],
                    root,
                    {"patch_size": 4, "stride": 2, "mask_threshold": 127},
                    segmentation_mode="multiclass",
                    merge_join_masks=False,
                    compute_soft_cldice_iterations=True,
                    iteration_margin=4,
                    iteration_round_up_to=2,
                )
                cached_model = _BatchRecordingIdentity()
                cached_loss = _BatchRecordingCrossEntropy()
                trainer.model = cached_model
                trainer.loss_fn = cached_loss
                trainer.full_image_batch_size = 8
                trainer.soft_cldice_iterations = None
                trainer.validation_patch_cache = validation_cache
                with patch("src.engine.trainer.Image.open") as image_open:
                    cached_metrics = trainer._evaluate_full_images(
                        epoch=1, epochs=1
                    )
                    image_open.assert_not_called()
                skeletonize_call_count = skeletonize_mock.call_count
                validation_cache.cleanup()

        self.assertEqual(batched_model.batch_sizes, [8, 1])
        self.assertEqual(batched_loss.batch_sizes, [3, 1])
        self.assertEqual(metrics["val_loss_patch_count"], 4)
        self.assertEqual(single_patch_model.batch_sizes, [1] * 9)
        self.assertEqual(single_patch_loss.batch_sizes, [1] * 4)
        self.assertEqual(cached_model.batch_sizes, [8, 1])
        self.assertEqual(cached_loss.batch_sizes, [3, 1])
        self.assertEqual(cached_loss.iteration_values, [[4] * 3, [4]])
        self.assertEqual(batched_loss.iteration_values, [[7] * 3, [7]])
        self.assertTrue(
            all(shape is not None for shape in batched_loss.geometry_weight_shapes)
        )
        self.assertEqual(metrics.keys(), single_patch_metrics.keys())
        self.assertEqual(metrics.keys(), cached_metrics.keys())
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                self.assertAlmostEqual(metric_value, single_patch_metrics[metric_name])
                self.assertAlmostEqual(metric_value, cached_metrics[metric_name])
        self.assertEqual(skeletonize_call_count, 4)

        expected = multiclass_metrics_from_masks(
            torch.from_numpy(prediction.astype(np.int64)),
            torch.from_numpy(target.astype(np.int64)),
            {"loci": 1, "inoculum": 2},
        )
        expected_dice_foreground = 0.5 * (
            expected["dice_loci"] + expected["dice_inoculum"]
        )
        self.assertAlmostEqual(
            metrics["val_dice_macro_foreground_per_image"],
            expected["dice_macro_foreground"],
        )
        expected_cldice = expected["cldice_loci"]
        self.assertAlmostEqual(metrics["val_cldice_loci_per_image"], expected_cldice)
        self.assertAlmostEqual(
            metrics["val_dice_cldice_per_image"],
            0.7 * expected_dice_foreground + 0.3 * expected_cldice,
        )
        self.assertAlmostEqual(
            metrics["val_dice_low_cldice_per_image"],
            0.9 * expected_dice_foreground + 0.1 * expected_cldice,
        )
        self.assertAlmostEqual(
            metrics["val_inoculum_compensated_per_image"],
            0.3 * expected["dice_loci"]
            + 0.5 * expected["dice_inoculum"]
            + 0.2 * expected_cldice,
        )


    def test_binary_full_image_validation_reuses_each_forward_for_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = np.zeros((4, 4), dtype=np.uint8)
            target[1, 1:3] = 255
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            image[..., 0] = target
            image_path = root / "binary.png"
            mask_path = root / "binary-mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(target).save(mask_path)
            record = OriginalImageRecord(
                "binary.png",
                image_path,
                mask_path,
                4,
                4,
            )

            trainer = Trainer.__new__(Trainer)
            model = _BatchRecordingFirstChannel()
            loss = _BatchRecordingBCE()
            trainer.model = model
            trainer.loss_fn = loss
            trainer.device = torch.device("cpu")
            trainer.data_config = {
                "patch_size": 2,
                "stride": 1,
                "mask_threshold": 127,
            }
            trainer.val_patch_transforms = self._raw_transform
            trainer.val_original_records = [record]
            trainer.segmentation_mode = "binary"
            trainer.segmentation_config = {"mode": "binary", "target": "loci"}
            trainer.class_names = {}
            trainer.threshold = 0.75
            trainer.use_amp = False
            trainer.use_tqdm = False
            trainer.fold_index = 0
            trainer.full_image_dice_weight = 0.5
            trainer.full_image_cldice_weight = 0.5
            trainer.full_image_batch_size = 4
            trainer.target_weight_builder = None
            trainer.soft_cldice_iterations = {"binary.png": 4}
            trainer.default_soft_cldice_iterations = 2

            metrics = trainer._evaluate_full_images(epoch=1, epochs=1)

        self.assertEqual(model.batch_sizes, [4, 4, 1])
        self.assertEqual(loss.batch_sizes, [2, 1, 1])
        self.assertEqual(loss.iteration_values, [[4, 4], [4], [4]])
        self.assertEqual(loss.target_foreground_counts, [[1, 1], [0], [0]])
        self.assertEqual(metrics["val_loss_patch_count"], 4)
        self.assertTrue(math.isfinite(metrics["val_loss"]))
        self.assertEqual(metrics["val_dice_per_image"], 1.0)
        self.assertEqual(metrics["val_iou_per_image"], 1.0)
        self.assertEqual(metrics["val_cldice_per_image"], 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            cached_image_path = cache_root / "binary.png"
            cached_mask_path = cache_root / "binary-mask.png"
            Image.fromarray(image).save(cached_image_path)
            Image.fromarray(target).save(cached_mask_path)
            cached_record = OriginalImageRecord(
                "binary.png",
                cached_image_path,
                cached_mask_path,
                4,
                4,
            )
            cache = build_static_validation_patch_cache(
                [cached_record],
                cache_root,
                {"patch_size": 2, "stride": 1, "mask_threshold": 127},
                segmentation_mode="binary",
                merge_join_masks=False,
                compute_soft_cldice_iterations=True,
                iteration_margin=4,
                iteration_round_up_to=2,
            )
            cached_model = _BatchRecordingFirstChannel()
            cached_loss = _BatchRecordingBCE()
            trainer.model = cached_model
            trainer.loss_fn = cached_loss
            trainer.val_original_records = [cached_record]
            trainer.soft_cldice_iterations = None
            trainer.validation_patch_cache = cache
            with patch("src.engine.trainer.Image.open") as image_open:
                cached_metrics = trainer._evaluate_full_images(
                    epoch=1, epochs=1
                )
                image_open.assert_not_called()

            self.assertEqual(cached_model.batch_sizes, [4, 4, 1])
            self.assertEqual(
                cached_loss.iteration_values,
                [[4, 4], [4], [4]],
            )
            self.assertEqual(metrics.keys(), cached_metrics.keys())
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    self.assertAlmostEqual(
                        metric_value, cached_metrics[metric_name]
                    )
            cache.cleanup()

    def test_reduce_on_plateau_consumes_validation_loss(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_scheduler(
            optimizer,
            {
                "name": "reduce_on_plateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 7,
                "min_lr": 1e-6,
                "monitor": "val_loss",
            },
        )
        trainer = Trainer.__new__(Trainer)
        trainer.scheduler = scheduler
        trainer.scheduler_monitor = "val_loss"

        trainer._step_scheduler(
            {
                "val_loss": 0.42,
                "val_dice_per_image": 0.99,
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
                    "monitor": "val_dice_per_image",
                    "enable_per_image_validation": True,
                    "per_image_validation_interval": 10,
                },
                loss_config=None,
                logger=None,
                fold_dir=Path("unused"),
                data_config={},
                val_original_records=[object()],
            )

    def test_best_checkpoint_selection_uses_full_image_dice(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.monitor = "val_dice_per_image"
        trainer.monitor_mode = "max"
        history = [
            {
                "epoch": 1,
                "val_dice_per_patch": 0.95,
                "val_dice_per_image": 0.40,
            },
            {
                "epoch": 2,
                "val_dice_per_patch": 0.70,
                "val_dice_per_image": 0.80,
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

    def test_full_image_validation_uses_half_patch_stride(self) -> None:
        shared = {
            "patch_size": 4,
            "overlap": 0,
            "stride": 4,
            "filter_empty_patches": False,
            "mask_threshold": 127,
            "min_foreground_pixels": 2,
        }

        effective = build_full_image_validation_patching_config(shared)

        self.assertEqual(effective["stride"], 2)
        self.assertEqual(effective["overlap"], 2)
        self.assertEqual(effective["filter_empty_patches"], False)

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
                self.assertNotIn("fast", config["validation"])
                primary_name = config["checkpointing"]["primary"]
                primary = config["checkpointing"]["selections"][primary_name]
                scheduler_monitor = str(config["scheduler"]["monitor"])
                self.assertGreaterEqual(config["validation"]["start_epoch"], 1)
                self.assertGreater(
                    config["validation"]["full_image"]["batch_size"], 0
                )
                self.assertEqual(scheduler_monitor, "val_dice_cldice_per_image")
                self.assertEqual(config["scheduler"]["mode"], "max")
                self.assertEqual(primary["monitor"], scheduler_monitor)
                self.assertEqual(primary["mode"], "max")
                self.assertEqual(
                    config["validation"]["full_image"]["loss"]["patch_selection"],
                    "non_overlapping",
                )
                if scheduler_monitor.endswith("_per_image"):
                    self.assertTrue(config["validation"]["full_image"]["enabled"])
                    self.assertEqual(
                        config["validation"]["full_image"]["interval_epochs"],
                        1,
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
            trainer.fit([object()], epochs=2)

        self.assertEqual([item[:2] for item in observed], [(1, 1), (2, 2)])
        self.assertTrue(all(manifest_count >= 2 for _, _, manifest_count in observed))

    def test_fit_cleans_validation_patch_cache(self) -> None:
        class _CleanupRecorder:
            def __init__(self) -> None:
                self.calls = 0

            def cleanup(self) -> None:
                self.calls += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _StubTrainer(Path(tmpdir), scores=[0.9])
            cache = _CleanupRecorder()
            trainer.validation_patch_cache = cache
            trainer.fit([object()], epochs=1)

        self.assertEqual(cache.calls, 1)

    def test_validation_start_epoch_delays_metrics_scheduler_and_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = Path(tmpdir)
            trainer = _StubTrainer(fold_dir, scores=[0.1, 0.2, 0.9])
            trainer.validation_start_epoch = 3
            trainer.save_last_checkpoint = False
            trainer.best_interval_checkpoint_enabled = False
            trainer.scheduler = build_scheduler(
                trainer.optimizer,
                {
                    "name": "reduce_on_plateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 7,
                    "min_lr": 0.001,
                    "monitor": "val_loss",
                },
            )

            result = trainer.fit([object()], epochs=3)
            checkpoint = torch.load(fold_dir / "best_current.pt", map_location="cpu")

        self.assertEqual(trainer.validation_epochs, [3])
        self.assertNotIn("val_loss", result["history"][0])
        self.assertNotIn("val_loss", result["history"][1])
        self.assertEqual(result["history"][2]["val_dice_per_image"], 0.9)
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(trainer.scheduler.last_epoch, 1)

    def test_fit_returns_metrics_from_one_consistent_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _StubTrainer(Path(tmpdir), scores=[0.9, 0.2])

            result = trainer.fit([object()], epochs=2)

        self.assertEqual(result["best_epoch"], 1)
        self.assertAlmostEqual(result["val_loss"], 0.1)
        self.assertEqual(result["val_dice_per_image"], 0.9)

    def test_best_only_checkpoint_setting_omits_last_and_interval_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = Path(tmpdir)
            trainer = _StubTrainer(fold_dir, scores=[0.9, 0.2])
            trainer.save_last_checkpoint = False
            trainer.best_interval_checkpoint_enabled = False

            trainer.fit([object()], epochs=2)

            checkpoints = sorted(path.name for path in fold_dir.glob("*.pt"))
            with (fold_dir / "checkpoint_manifest.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))

        self.assertEqual(
            checkpoints,
            [
                "best_current.pt",
                "best_dice.pt",
                "best_inoculum_compensated.pt",
                "best_low_cldice.pt",
            ],
        )
        self.assertEqual(len(manifest_rows), 4)
        self.assertEqual(
            {row["checkpoint"] for row in manifest_rows},
            set(checkpoints),
        )

    def test_each_best_checkpoint_tracks_its_own_monitor(self) -> None:
        validation_metrics = [
            {
                "val_loss": 0.8,
                "val_dice_cldice_per_image": 0.9,
                "val_dice_low_cldice_per_image": 0.1,
                "val_inoculum_compensated_per_image": 0.2,
            },
            {
                "val_loss": 0.1,
                "val_dice_cldice_per_image": 0.5,
                "val_dice_low_cldice_per_image": 0.95,
                "val_inoculum_compensated_per_image": 0.3,
            },
            {
                "val_loss": 0.5,
                "val_dice_cldice_per_image": 0.4,
                "val_dice_low_cldice_per_image": 0.2,
                "val_inoculum_compensated_per_image": 0.99,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = Path(tmpdir)
            trainer = _StubTrainer(
                fold_dir,
                scores=[0.9, 0.5, 0.4],
                validation_metrics=validation_metrics,
            )
            trainer.save_last_checkpoint = False
            trainer.best_interval_checkpoint_enabled = False

            trainer.fit([object()], epochs=3)
            selected_epochs = {
                path.name: int(torch.load(path, map_location="cpu")["epoch"])
                for path in fold_dir.glob("best_*.pt")
            }

        self.assertEqual(selected_epochs, {
            "best_current.pt": 1,
            "best_dice.pt": 1,
            "best_low_cldice.pt": 2,
            "best_inoculum_compensated.pt": 3,
        })

    def test_last_checkpoint_contains_post_scheduler_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _StubTrainer(Path(tmpdir), scores=[1.0, 0.5])
            trainer.scheduler = build_scheduler(
                trainer.optimizer,
                {
                    "name": "reduce_on_plateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 0,
                    "min_lr": 0.001,
                },
            )

            trainer.fit([object()], epochs=2)
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
                "non-finite metric 'val_loss'",
            ):
                trainer.fit([object()], epochs=1)

            self.assertFalse(any(fold_dir.glob("best_*.pt")))
            self.assertFalse((fold_dir / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
