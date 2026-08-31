from __future__ import annotations

import unittest

import torch

from src.losses.combined import (
    MulticlassCEDiceLociCLDiceLoss,
    MulticlassGeometryCEDiceLociCLDiceLoss,
)
from src.losses.factory import build_loss
from src.metrics.segmentation import multiclass_metrics_from_masks
from src.models.factory import build_model
from src.models.wrappers import extract_logits
from src.optim.factory import build_optimizer
from src.utils.config import load_config


class ModelFactoryTests(unittest.TestCase):
    def test_multiclass_segformer_b2_refinement_pipeline(self) -> None:
        config = load_config("multiclass-segformer-mit-b2-refinement-config.yaml")

        self.assertEqual(config["model"]["name"], "segformer_mit_b2_refinement")
        self.assertEqual(config["model"]["encoder_name"], "mit_b2")
        self.assertEqual(config["model"]["encoder_weights"], "imagenet")
        self.assertEqual(config["model"]["shallow_channels"], [16, 32])
        self.assertEqual(config["model"]["refine_half_channels"], [128, 64])
        self.assertEqual(config["model"]["refine_full_channels"], [32, 32])

        config["model"]["encoder_weights"] = None
        model = build_model(config["model"]).eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 64, 64))

        self.assertEqual(
            tuple(model.encoder.out_channels[-4:]),
            (64, 128, 320, 512),
        )
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))

    def test_multiclass_segformer_b1_refinement_pipeline(self) -> None:
        config = load_config("multiclass-segformer-mit-b1-refinement-config.yaml")

        self.assertEqual(config["segmentation"]["mode"], "multiclass")
        self.assertEqual(
            config["segmentation"]["classes"],
            {"background": 0, "loci": 1, "inoculum": 2},
        )
        self.assertEqual(config["patching"]["patch_size"], 1024)
        self.assertEqual(config["model"]["name"], "segformer_mit_b1_refinement")
        self.assertEqual(config["model"]["encoder_name"], "mit_b1")
        self.assertEqual(config["model"]["decoder_segmentation_channels"], 256)
        self.assertEqual(config["model"]["shallow_channels"], [16, 32])
        self.assertEqual(config["model"]["refine_half_channels"], [128, 64])
        self.assertEqual(config["model"]["refine_full_channels"], [32, 32])

        config["model"]["encoder_weights"] = None
        model = build_model(config["model"]).eval()
        captured: dict[str, object] = {}
        model.encoder.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("encoder", output)
        )
        for name in (
            "decoder",
            "shallow_full",
            "shallow_half",
            "refine_half",
            "refine_full",
        ):
            getattr(model, name).register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.__setitem__(
                    name, tuple(output.shape)
                )
            )

        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 64, 64))

        encoder_features = captured["encoder"]
        self.assertIsInstance(encoder_features, list)
        self.assertEqual(
            [tuple(feature.shape) for feature in encoder_features[-4:]],
            [
                (1, 64, 16, 16),
                (1, 128, 8, 8),
                (1, 320, 4, 4),
                (1, 512, 2, 2),
            ],
        )
        self.assertEqual(captured["decoder"], (1, 256, 16, 16))
        self.assertEqual(captured["shallow_full"], (1, 16, 64, 64))
        self.assertEqual(captured["shallow_half"], (1, 32, 32, 32))
        self.assertEqual(captured["refine_half"], (1, 64, 32, 32))
        self.assertEqual(captured["refine_full"], (1, 32, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))

        expected_refinement_types = [
            torch.nn.Conv2d,
            torch.nn.BatchNorm2d,
            torch.nn.GELU,
            torch.nn.Conv2d,
            torch.nn.BatchNorm2d,
            torch.nn.GELU,
        ]
        self.assertEqual(
            [type(module) for module in model.refine_half],
            expected_refinement_types,
        )
        self.assertEqual(
            [type(module) for module in model.refine_full],
            expected_refinement_types,
        )
        self.assertEqual(
            (model.refine_half[0].in_channels, model.refine_half[0].out_channels),
            (288, 128),
        )
        self.assertEqual(
            (model.refine_half[3].in_channels, model.refine_half[3].out_channels),
            (128, 64),
        )
        self.assertEqual(
            (model.refine_full[0].in_channels, model.refine_full[0].out_channels),
            (80, 32),
        )
        self.assertEqual(
            (model.refine_full[3].in_channels, model.refine_full[3].out_channels),
            (32, 32),
        )

        optimizer = build_optimizer(model, config["optimizer"])
        groups = {group["group_name"]: group for group in optimizer.param_groups}
        encoder_ids = {id(parameter) for parameter in groups["encoder"]["params"]}
        decoder_ids = {id(parameter) for parameter in groups["decoder"]["params"]}
        self.assertEqual(groups["encoder"]["lr"], 1.0e-5)
        self.assertEqual(groups["decoder"]["lr"], 1.0e-4)
        self.assertEqual(
            encoder_ids,
            {id(parameter) for parameter in model.encoder.parameters()},
        )
        for module in (
            model.decoder,
            model.shallow_full,
            model.shallow_half,
            model.refine_half,
            model.refine_full,
            model.segmentation_head,
        ):
            self.assertTrue(
                {id(parameter) for parameter in module.parameters()} <= decoder_ids
            )
        self.assertFalse(encoder_ids & decoder_ids)
        self.assertEqual(
            encoder_ids | decoder_ids,
            {id(parameter) for parameter in model.parameters()},
        )

    def test_segformer_b1_refinement_rejects_invalid_channel_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "shallow_channels"):
            build_model(
                {
                    "name": "segformer_mit_b1_refinement",
                    "encoder_weights": None,
                    "shallow_channels": [16],
                }
            )

    def test_refinement_model_name_rejects_mismatched_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires encoder_name='mit_b2'"):
            build_model(
                {
                    "name": "segformer_mit_b2_refinement",
                    "encoder_name": "mit_b1",
                    "encoder_weights": None,
                }
            )

    def test_builds_segformer_mit_b3(self) -> None:
        config = load_config("config_segformer_mit_b3.yaml")
        config["model"]["encoder_weights"] = None

        model = build_model(config["model"])
        model.eval()

        with torch.no_grad():
            logits = extract_logits(model(torch.zeros(1, 3, 64, 64)))

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))

    def test_multiclass_segformer_b3_geometry_config_builds_pipeline(self) -> None:
        config = load_config("multiclass-segformer-mit-b3-geometry-config.yaml")
        self.assertEqual(config["model"]["encoder_name"], "mit_b3")
        self.assertEqual(config["model"]["num_classes"], 3)
        self.assertEqual(config["optimizer"]["encoder_lr"], 1.0e-5)
        self.assertEqual(config["optimizer"]["decoder_lr"], 1.0e-4)
        self.assertEqual(
            config["validation"]["full_image"],
            {
                "enabled": False,
                "batch_size": 1,
                "interval_epochs": 1,
                "selection": "smallest_area",
                "max_images": 3,
                "monitor": {"dice_weight": 0.5, "cldice_weight": 0.5},
            },
        )

        config["model"]["encoder_weights"] = None
        model = build_model(config["model"]).eval()
        with torch.no_grad():
            logits = extract_logits(model(torch.zeros(1, 3, 64, 64)))
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))
        self.assertIsInstance(
            build_loss(config["loss"]),
            MulticlassGeometryCEDiceLociCLDiceLoss,
        )

    def test_multiclass_segformer_b5_config_builds_pipeline(self) -> None:
        config = load_config("multiclass-segformer-config.yaml")

        self.assertEqual(config["segmentation"]["mode"], "multiclass")
        self.assertEqual(config["project"]["name"], "fungi_multiclass_segmentation_segformer_mit_b5")
        self.assertEqual(config["model"]["name"], "segformer_mit_b5")
        self.assertEqual(config["model"]["encoder_name"], "mit_b5")
        self.assertEqual(config["model"]["num_classes"], 3)
        self.assertGreater(int(config["train"]["epochs"]), 0)
        self.assertEqual(
            config["scheduler"]["monitor"],
            config["train"]["monitor"],
        )
        self.assertEqual(
            set(config["scheduler"]["min_lr"]),
            {"encoder", "decoder"},
        )

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
