from __future__ import annotations

import unittest

import torch

from src.losses.combined import MulticlassCEDiceLociCLDiceLoss
from src.losses.factory import build_loss
from src.metrics.segmentation import multiclass_metrics_from_masks
from src.models.factory import build_model
from src.models.wrappers import extract_logits
from src.utils.config import load_config


class ModelFactoryTests(unittest.TestCase):
    def test_builds_segformer_mit_b3(self) -> None:
        config = load_config("config_segformer_mit_b3.yaml")
        config["model"]["encoder_weights"] = None

        model = build_model(config["model"])
        model.eval()

        with torch.no_grad():
            logits = extract_logits(model(torch.zeros(1, 3, 64, 64)))

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))

    def test_multiclass_segformer_config_matches_reference_and_builds_pipeline(self) -> None:
        reference = load_config("multiclass-config.yaml")
        config = load_config("multiclass-segformer-config.yaml")

        for section in reference.keys() - {"project", "model"}:
            self.assertEqual(config[section], reference[section], section)

        self.assertEqual(config["segmentation"]["mode"], "multiclass")
        self.assertEqual(config["model"]["name"], "segformer_mit_b3")
        self.assertEqual(config["model"]["num_classes"], 3)
        self.assertEqual(config["train"]["monitor"], "val_dice_macro_foreground")
        self.assertEqual(config["scheduler"]["monitor"], "val_dice_macro_foreground")
        self.assertFalse(config["test_evaluation"]["threshold_sweep"])
        self.assertTrue(config["qualitative_evaluation"]["enabled"])

        config["model"]["encoder_weights"] = None
        model = build_model(config["model"]).eval()
        with torch.no_grad():
            logits = extract_logits(model(torch.zeros(1, 3, 64, 64)))
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))

        loss = build_loss(config["loss"])
        self.assertIsInstance(loss, MulticlassCEDiceLociCLDiceLoss)
        targets = torch.zeros((1, 64, 64), dtype=torch.long)
        targets[:, 16:48, 28:36] = 1
        targets[:, :16, :16] = 2
        self.assertTrue(torch.isfinite(loss(logits, targets)))

        predictions = logits.softmax(dim=1).argmax(dim=1)
        metrics = multiclass_metrics_from_masks(predictions, targets)
        self.assertEqual(
            set(metrics),
            {
                "dice_loci", "iou_loci", "precision_loci", "recall_loci",
                "dice_inoculum", "iou_inoculum", "precision_inoculum", "recall_inoculum",
                "dice_macro_foreground", "iou_macro_foreground", "cldice_loci",
            },
        )


if __name__ == "__main__":
    unittest.main()
