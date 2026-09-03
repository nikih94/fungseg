from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.engine.trainer import Trainer
from src.losses.combined import MulticlassCEDiceLociCLDiceLoss
from src.utils.run_resume import contiguous_completed_folds


class _CountingLoader:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter([self.batch])

    def __len__(self) -> int:
        return 1


class _Progress:
    def __init__(self, iterable) -> None:
        self.iterable = iterable

    def __iter__(self):
        return self.iterable

    def set_postfix(self, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class TrainingPerformanceTests(unittest.TestCase):
    def test_tqdm_wraps_one_existing_loader_iterator(self) -> None:
        loader = _CountingLoader({
            "image": torch.zeros(1, 3, 4, 4),
            "mask": torch.zeros(1, 4, 4, dtype=torch.long),
        })
        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Conv2d(3, 3, 1)
        trainer.loss_fn = MulticlassCEDiceLociCLDiceLoss(iterations=1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
        trainer.device = torch.device("cpu")
        trainer.use_amp = False
        trainer.use_tqdm = True
        trainer.fold_index = 0
        trainer.segmentation_mode = "multiclass"
        trainer.class_names = {"loci": 1, "inoculum": 2}
        trainer.compute_hard_cldice_metrics = False
        trainer.loss_config = {"name": "multiclass_ce_dice_loci_cldice"}
        trainer.train_config = {"grad_clip": None}

        with (
            patch("src.engine.trainer.tqdm", side_effect=lambda iterable, **_: _Progress(iterable)),
            patch("src.metrics.segmentation.skeletonize") as skeletonize_mock,
        ):
            metrics = trainer._run_epoch(loader, training=True, epoch=1, epochs=1)

        self.assertEqual(loader.iter_calls, 1)
        skeletonize_mock.assert_not_called()
        self.assertNotIn("train_cldice_loci", metrics)
        self.assertIn("train_iou_loci", metrics)
        self.assertIn("train_precision_loci", metrics)
        self.assertIn("train_recall_loci", metrics)

    def test_multiclass_training_reuses_computed_loss_components(self) -> None:
        loss = MulticlassCEDiceLociCLDiceLoss(iterations=1)
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        targets = torch.zeros(2, 8, 8, dtype=torch.long)
        targets[0, 3, 2:6] = 1
        with patch(
            "src.losses.combined.soft_cldice_scores_from_probabilities",
            wraps=__import__(
                "src.losses.combined", fromlist=["soft_cldice_scores_from_probabilities"]
            ).soft_cldice_scores_from_probabilities,
        ) as score_mock:
            total, parts = loss.forward_with_components(logits, targets)
            total.backward()

        self.assertEqual(score_mock.call_count, 1)
        self.assertEqual(set(parts), {"cross_entropy", "dice", "loci_cldice"})

    def test_foreground_filter_normalizes_over_original_batch(self) -> None:
        loss = MulticlassCEDiceLociCLDiceLoss(iterations=1)
        logits = torch.zeros(2, 3, 4, 4)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)
        targets[0, 1, 1] = 1
        with patch(
            "src.losses.combined.soft_cldice_scores_from_probabilities",
            return_value=torch.tensor([0.25]),
        ) as score_mock:
            parts = loss.components(
                logits, targets, soft_cldice_sample_mask=torch.tensor([True, False])
            )
        self.assertAlmostEqual(parts["loci_cldice"].item(), 0.375)
        self.assertEqual(score_mock.call_args.args[0].shape[0], 1)


    def test_completed_fold_inference_stops_at_first_gap(self) -> None:
        rows = [{"fold": "0"}, {"fold": "1"}, {"fold": "3"}]
        self.assertEqual(contiguous_completed_folds(rows, 5), [0, 1])


if __name__ == "__main__":
    unittest.main()
