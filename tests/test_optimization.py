from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from torch import nn

from src.engine.trainer import Trainer
from src.optim.factory import build_optimizer
from src.schedulers.factory import build_scheduler
from src.utils.config import load_config


class ToySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.segmentation_head = nn.Linear(4, 1)


class OptimizerFactoryTests(unittest.TestCase):
    def test_builds_disjoint_encoder_and_decoder_parameter_groups(self) -> None:
        model = ToySegmentationModel()
        optimizer = build_optimizer(
            model,
            {
                "name": "adamw",
                "encoder_lr": 1.0e-5,
                "decoder_lr": 1.0e-4,
                "weight_decay": 1.0e-4,
            },
        )

        groups = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"encoder", "decoder"})
        self.assertEqual(groups["encoder"]["lr"], 1.0e-5)
        self.assertEqual(groups["decoder"]["lr"], 1.0e-4)

        encoder_ids = {id(parameter) for parameter in groups["encoder"]["params"]}
        decoder_ids = {id(parameter) for parameter in groups["decoder"]["params"]}
        self.assertEqual(encoder_ids, {id(parameter) for parameter in model.encoder.parameters()})
        self.assertEqual(
            decoder_ids,
            {
                id(parameter)
                for module in (model.decoder, model.segmentation_head)
                for parameter in module.parameters()
            },
        )
        self.assertFalse(encoder_ids & decoder_ids)
        self.assertEqual(encoder_ids | decoder_ids, {id(parameter) for parameter in model.parameters()})

    def test_preserves_single_learning_rate_configuration(self) -> None:
        model = ToySegmentationModel()
        optimizer = build_optimizer(
            model.parameters(),
            {"name": "adamw", "lr": 2.0e-4, "weight_decay": 1.0e-4},
        )

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2.0e-4)
        self.assertNotIn("group_name", optimizer.param_groups[0])

    def test_rejects_partial_split_learning_rate_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be configured together"):
            build_optimizer(
                ToySegmentationModel(),
                {"name": "adamw", "encoder_lr": 1.0e-5},
            )

    def test_rejects_split_learning_rates_for_model_without_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "model with an 'encoder' module"):
            build_optimizer(
                nn.Linear(4, 1),
                {
                    "name": "adamw",
                    "encoder_lr": 1.0e-5,
                    "decoder_lr": 1.0e-4,
                },
            )


class SchedulerFactoryTests(unittest.TestCase):
    def _build_split_optimizer(self):
        return build_optimizer(
            ToySegmentationModel(),
            {
                "name": "adamw",
                "encoder_lr": 1.0e-5,
                "decoder_lr": 1.0e-4,
            },
        )

    def test_reduce_on_plateau_updates_both_groups_and_respects_named_floors(self) -> None:
        optimizer = self._build_split_optimizer()
        scheduler = build_scheduler(
            optimizer,
            {
                "name": "reduce_on_plateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 0,
                "min_lr": {"encoder": 1.0e-7, "decoder": 1.0e-6},
            },
        )

        scheduler.step(1.0)
        scheduler.step(1.0)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [5.0e-6, 5.0e-5])

        for _ in range(20):
            scheduler.step(1.0)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1.0e-7, 1.0e-6])

    def test_named_floors_must_match_optimizer_groups(self) -> None:
        optimizer = self._build_split_optimizer()
        with self.assertRaisesRegex(ValueError, "must match optimizer groups"):
            build_scheduler(
                optimizer,
                {
                    "name": "reduce_on_plateau",
                    "min_lr": {"encoder": 1.0e-7},
                },
            )


class LearningRateConfigAndReportingTests(unittest.TestCase):
    def test_differential_learning_rate_configs_do_not_retain_inherited_single_lr(self) -> None:
        for config_path in (
            "config_segformer_mit_b3.yaml",
            "multiclass-config.yaml",
            "multiclass-segformer-mit-b1-refinement-config.yaml",
            "multiclass-segformer-mit-b2-refinement-config.yaml",
            "multiclass-segformer-config.yaml",
            "multiclass-segformer-mit-b3-geometry-config.yaml",
        ):
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertNotIn("lr", config["optimizer"])
                self.assertGreater(float(config["optimizer"]["encoder_lr"]), 0.0)
                self.assertGreater(float(config["optimizer"]["decoder_lr"]), 0.0)
                self.assertEqual(set(config["scheduler"]["min_lr"]), {"encoder", "decoder"})
                self.assertLessEqual(
                    float(config["scheduler"]["min_lr"]["encoder"]),
                    float(config["optimizer"]["encoder_lr"]),
                )
                self.assertLessEqual(
                    float(config["scheduler"]["min_lr"]["decoder"]),
                    float(config["optimizer"]["decoder_lr"]),
                )

    def test_config_rejects_partial_split_learning_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "optimizer:\n  name: adamw\n  encoder_lr: 1.0e-5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be configured together"):
                load_config(config_path)

    def test_trainer_reports_both_rates_and_keeps_decoder_alias(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.optimizer = build_optimizer(
            ToySegmentationModel(),
            {
                "name": "adamw",
                "encoder_lr": 1.0e-5,
                "decoder_lr": 1.0e-4,
            },
        )

        self.assertEqual(
            trainer._current_learning_rates(),
            {"lr": 1.0e-4, "encoder_lr": 1.0e-5, "decoder_lr": 1.0e-4},
        )


if __name__ == "__main__":
    unittest.main()
