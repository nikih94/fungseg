from __future__ import annotations

import unittest

import numpy as np
import torch
from skimage.morphology import skeletonize

from src.engine.trainer import Trainer
from src.losses.combined import MulticlassCEDiceLociCLDiceLoss
from src.losses.factory import build_loss
from src.metrics.segmentation import (
    cldice_score_from_masks,
    hard_skeletonize_masks,
    multiclass_metrics_from_masks,
    precision_score_from_masks,
    recall_score_from_masks,
    soft_cldice_scores_from_probabilities,
)
from src.utils.config import config_for_persistence


class MetricAndLossRegressionTests(unittest.TestCase):
    def test_hard_skeletonizer_matches_skimage_zhang_for_batched_masks(self) -> None:
        branch = np.zeros((64, 64), dtype=bool)
        branch[8:56, 29:35] = True
        branch[28:36, 8:56] = True
        loop = np.zeros_like(branch)
        loop[8:56, 8:14] = True
        loop[8:14, 8:56] = True
        loop[50:56, 8:56] = True
        loop[8:56, 50:56] = True
        masks = np.stack([branch, loop], axis=0)

        actual = hard_skeletonize_masks(torch.from_numpy(masks)).squeeze(1).numpy()
        expected = np.stack(
            [skeletonize(mask, method="zhang") for mask in masks],
            axis=0,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_cldice_harmonic_mean_penalizes_disjoint_skeletons(self) -> None:
        target = torch.zeros(1, 1, 32, 160)
        prediction = torch.zeros_like(target)
        target[:, :, 10, 16:144] = 1.0
        prediction[:, :, 20, 16:144] = 1.0

        soft_score = soft_cldice_scores_from_probabilities(
            prediction, target, iterations=3, smooth=1.0
        ).item()
        hard_score = cldice_score_from_masks(prediction, target)

        self.assertLess(soft_score, 0.02)
        self.assertLess(hard_score, 1e-4)
        self.assertAlmostEqual(cldice_score_from_masks(target, target), 1.0, places=6)

    def test_hard_cldice_skeletonizes_thick_masks_with_zhang(self) -> None:
        target = torch.zeros(1, 64, 64)
        prediction = torch.zeros_like(target)
        target[:, :16, :16] = 1.0
        prediction[:, -16:, -16:] = 1.0

        self.assertLess(cldice_score_from_masks(prediction, target), 1e-4)

        full_frame = torch.ones(1, 32, 32)
        centered = torch.zeros_like(full_frame)
        centered[:, 8:24, 8:24] = 1.0
        self.assertAlmostEqual(
            cldice_score_from_masks(centered, full_frame), 1.0, places=6
        )

    def test_every_canonical_binary_loss_has_finite_gradients(self) -> None:
        configs = [
            {"name": "bce"},
            {"name": "bce_dice", "smooth": 1e-5},
            {"name": "bce_dice_cldice", "iterations": 3},
            {"name": "tversky"},
            {"name": "cldice", "iterations": 3},
            {"name": "soft_cldice", "iterations": 3},
            {"name": "tversky_soft_cldice", "iterations": 3},
        ]
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 3:13, 7:9] = 1.0

        for config in configs:
            with self.subTest(loss=config["name"]):
                torch.manual_seed(11)
                logits = torch.randn(2, 1, 16, 16, requires_grad=True)
                loss = build_loss(config)(logits, target)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(logits.grad)
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_multiclass_loci_cldice_penalizes_disjoint_lines(self) -> None:
        target = torch.zeros(1, 32, 160, dtype=torch.long)
        target[:, 10, 16:144] = 1
        logits = torch.full((1, 3, 32, 160), -12.0)
        logits[:, 0] = 12.0
        logits[:, 0, 20, 16:144] = -12.0
        logits[:, 1, 20, 16:144] = 12.0

        components = MulticlassCEDiceLociCLDiceLoss(iterations=3).components(logits, target)

        self.assertGreater(components["loci_cldice"].item(), 0.98)

    def test_multiclass_loss_components_promote_large_half_logits_to_float32(self) -> None:
        logits = torch.zeros((1, 3, 256, 256), dtype=torch.float16)
        target = torch.zeros((1, 256, 256), dtype=torch.long)

        components = MulticlassCEDiceLociCLDiceLoss(iterations=0).components(
            logits, target
        )

        self.assertEqual(components["cross_entropy"].dtype, torch.float32)
        for component in components.values():
            self.assertTrue(torch.isfinite(component))

    def test_multiclass_macro_excludes_only_empty_empty_classes(self) -> None:
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, 2:6, 2:6] = 1
        prediction = torch.zeros_like(target)

        metrics = multiclass_metrics_from_masks(prediction, target)

        self.assertLess(metrics["dice_macro_foreground"], 1e-5)
        self.assertAlmostEqual(metrics["dice_inoculum"], 1.0)
        empty_metrics = multiclass_metrics_from_masks(
            torch.zeros_like(target), torch.zeros_like(target)
        )
        self.assertAlmostEqual(empty_metrics["dice_macro_foreground"], 1.0)

    def test_precision_and_recall_handle_one_sided_empty_masks(self) -> None:
        empty = torch.zeros(8, 8)
        foreground = torch.ones(8, 8)

        self.assertEqual(precision_score_from_masks(empty, foreground), 0.0)
        self.assertEqual(recall_score_from_masks(foreground, empty), 0.0)
        self.assertEqual(precision_score_from_masks(empty, empty), 1.0)
        self.assertEqual(recall_score_from_masks(empty, empty), 1.0)

    def test_bce_dice_factory_honors_smooth(self) -> None:
        loss = build_loss({"name": "bce_dice", "smooth": 7.0})
        self.assertEqual(loss.smooth, 7.0)

    def test_persisted_loss_config_tracks_only_effective_options(self) -> None:
        base = {
            "segmentation": {"mode": "binary", "target": "legacy"},
            "paths": {},
            "model": {"name": "unetplusplus_resnet34"},
            "scheduler": {"name": "none"},
            "split": {"mode": "train_val"},
        }
        cldice_config = {
            **base,
            "loss": {
                "name": "cldice",
                "threshold": 0.9,
                "iterations": 4,
                "cldice_smooth": 1.0,
                "dice_weight": 0.7,
            },
        }
        bce_dice_config = {
            **base,
            "loss": {
                "name": "bce_dice",
                "bce_weight": 0.4,
                "dice_weight": 0.6,
                "smooth": 0.25,
                "iterations": 9,
            },
        }

        persisted_cldice = config_for_persistence(cldice_config)
        persisted_bce_dice = config_for_persistence(bce_dice_config)

        self.assertEqual(
            persisted_cldice["loss"],
            {"name": "cldice", "iterations": 4, "cldice_smooth": 1.0},
        )
        self.assertEqual(
            persisted_bce_dice["loss"],
            {
                "name": "bce_dice",
                "bce_weight": 0.4,
                "dice_weight": 0.6,
                "smooth": 0.25,
            },
        )

    def test_run_epoch_rejects_non_finite_loss_immediately(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Identity()
        trainer.loss_fn = lambda logits, _: logits.mean() * float("nan")
        trainer.device = torch.device("cpu")
        trainer.fold_index = 0
        trainer.use_tqdm = False
        trainer.use_amp = False

        with self.assertRaisesRegex(
            FloatingPointError, "val batch 1 produced non-finite loss"
        ):
            trainer._run_epoch(
                [{"image": torch.zeros(1, 1, 2, 2), "mask": torch.zeros(1, 1, 2, 2)}],
                training=False,
                epoch=1,
                epochs=1,
            )

    def test_epoch_aggregation_is_sample_weighted(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Identity()
        trainer.loss_fn = torch.nn.BCEWithLogitsLoss()
        trainer.device = torch.device("cpu")
        trainer.train_config = {}
        trainer.loss_config = {"name": "bce"}
        trainer.fold_index = 0
        trainer.segmentation_mode = "binary"
        trainer.threshold = 0.5
        trainer.use_tqdm = False
        trainer.use_amp = False

        perfect_logits = torch.full((2, 1, 2, 2), 20.0)
        perfect_targets = torch.ones_like(perfect_logits)
        bad_logits = torch.full((1, 1, 2, 2), -20.0)
        bad_targets = torch.ones_like(bad_logits)
        batches = [
            {"image": perfect_logits, "mask": perfect_targets},
            {"image": bad_logits, "mask": bad_targets},
        ]

        metrics = trainer._run_epoch(batches, training=False, epoch=1, epochs=1)
        perfect_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            perfect_logits, perfect_targets
        ).item()
        bad_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            bad_logits, bad_targets
        ).item()
        expected_loss = (2.0 * perfect_loss + bad_loss) / 3.0

        self.assertAlmostEqual(metrics["val_loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["val_dice_per_patch"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["val_iou_per_patch"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["val_bce"], expected_loss, places=6)


if __name__ == "__main__":
    unittest.main()
