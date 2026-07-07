from __future__ import annotations

import unittest

import torch

from src.metrics.loss_components import loss_component_metrics


class LossComponentMetricTests(unittest.TestCase):
    def test_perfect_prediction_has_high_soft_scores(self) -> None:
        targets = torch.zeros(1, 1, 16, 16)
        targets[:, :, 4:12, 8] = 1.0
        logits = torch.where(targets > 0, torch.full_like(targets, 20.0), torch.full_like(targets, -20.0))

        metrics = loss_component_metrics(
            logits,
            targets,
            {
                "name": "bce_dice_cldice",
                "bce_weight": 0.3,
                "dice_weight": 0.5,
                "soft_cldice_weight": 0.2,
                "iterations": 4,
                "smooth": 1e-6,
                "cldice_smooth": 1.0,
            },
        )

        self.assertLess(metrics["bce"], 1e-4)
        self.assertGreater(metrics["soft_dice_score"], 0.999)
        self.assertGreater(metrics["soft_cldice_score"], 0.999)
        self.assertAlmostEqual(metrics["weighted_bce"], metrics["bce"] * 0.3)
        self.assertAlmostEqual(metrics["weighted_soft_dice_loss"], (1.0 - metrics["soft_dice_score"]) * 0.5)
        self.assertAlmostEqual(metrics["weighted_soft_cldice_loss"], (1.0 - metrics["soft_cldice_score"]) * 0.2)

    def test_bad_prediction_has_lower_soft_scores(self) -> None:
        targets = torch.zeros(1, 1, 16, 16)
        targets[:, :, 4:12, 8] = 1.0
        logits = torch.where(targets > 0, torch.full_like(targets, -20.0), torch.full_like(targets, 20.0))

        metrics = loss_component_metrics(
            logits,
            targets,
            {
                "name": "bce_dice_cldice",
                "bce_weight": 0.3,
                "dice_weight": 0.5,
                "soft_cldice_weight": 0.2,
                "iterations": 4,
                "smooth": 1e-6,
                "cldice_smooth": 1.0,
            },
        )

        self.assertGreater(metrics["bce"], 10.0)
        self.assertLess(metrics["soft_dice_score"], 0.01)
        self.assertLess(metrics["soft_cldice_score"], 0.7)

    def test_tversky_metrics_are_reported_for_tversky_loss(self) -> None:
        targets = torch.zeros(1, 1, 8, 8)
        targets[:, :, 2:6, 2:6] = 1.0
        logits = torch.where(targets > 0, torch.full_like(targets, 20.0), torch.full_like(targets, -20.0))

        metrics = loss_component_metrics(
            logits,
            targets,
            {
                "name": "tversky_soft_cldice",
                "alpha": 0.3,
                "beta": 0.7,
                "tversky_weight": 0.7,
                "soft_cldice_weight": 0.3,
                "iterations": 3,
                "smooth": 1e-6,
                "cldice_smooth": 1.0,
            },
        )

        self.assertGreater(metrics["tversky_index"], 0.999)
        self.assertGreater(metrics["soft_cldice_score"], 0.999)
        self.assertAlmostEqual(metrics["weighted_tversky_loss"], (1.0 - metrics["tversky_index"]) * 0.7)


if __name__ == "__main__":
    unittest.main()
