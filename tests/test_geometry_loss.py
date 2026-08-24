from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import SegmentationPatchDataset
from src.engine.trainer import Trainer
from src.losses.combined import MulticlassGeometryCEDiceLociCLDiceLoss
from src.losses.factory import build_loss
from src.losses.geometry import GeometryWeightMapBuilder, build_geometry_weight_map_builder
from src.metrics.loss_components import loss_component_metrics
from src.patching import PatchRecord


class GeometryWeightMapTests(unittest.TestCase):
    def test_empty_target_has_unit_weights(self) -> None:
        target = np.zeros((16, 16), dtype=np.uint8)
        weights = GeometryWeightMapBuilder().build_numpy(target)
        np.testing.assert_array_equal(weights, np.ones_like(weights))

    def test_loci_center_is_weighted_more_than_border_without_reducing_pixels(self) -> None:
        target = np.zeros((25, 25), dtype=np.uint8)
        target[3:22, 7:18] = 1
        weights = GeometryWeightMapBuilder(separator_enabled=False).build_numpy(target)
        self.assertGreater(weights[12, 12], weights[12, 7])
        self.assertGreater(weights[12, 7], 1.0)
        self.assertGreaterEqual(float(weights.min()), 1.0)
        self.assertAlmostEqual(float(weights[12, 12]), 3.0, places=6)

    def test_separator_weights_only_annotated_background(self) -> None:
        target = np.zeros((25, 25), dtype=np.uint8)
        target[8:17, 3:10] = 1
        target[8:17, 12:19] = 1
        target[11:14, 10:12] = 2
        builder = GeometryWeightMapBuilder(
            center_multiplier=0.0,
            separator_multiplier=3.0,
            separator_radius_multipliers=(1.0,),
        )
        weights = builder.build_numpy(target)
        self.assertEqual(float(weights[12, 10]), 1.0)
        self.assertEqual(float(weights[12, 11]), 1.0)
        self.assertGreater(float(weights[10, 10]), 1.0)
        self.assertEqual(float(weights[12, 0]), 1.0)
        self.assertTrue(np.all(weights[target == 1] == 1.0))
        self.assertTrue(np.all(weights[target == 2] == 1.0))

    def test_factory_only_builds_weights_for_geometry_loss(self) -> None:
        self.assertIsNone(build_geometry_weight_map_builder({"name": "bce_dice"}))
        builder = build_geometry_weight_map_builder(
            {
                "name": "multiclass_geometry_ce_dice_loci_cldice",
                "geometry_aware_ce": {"separator_radius_multipliers": [0.5, 1.0]},
            }
        )
        self.assertIsInstance(builder, GeometryWeightMapBuilder)

    def test_dataset_builds_weights_from_transformed_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
            loci = np.zeros((4, 4), dtype=np.uint8)
            loci[0, 0] = 255
            Image.fromarray(loci).save(loci_path)
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(inoculum_path)
            record = PatchRecord(
                source_id="image.png",
                image_path=image_path,
                mask_path=loci_path,
                x=0,
                y=0,
                patch_size=4,
                scale=1.0,
                scaled_width=4,
                scaled_height=4,
                resolution_bucket="tiny",
                scale_label="normal",
                mask_paths={"loci": loci_path, "inoculum": inoculum_path},
            )
            seen: list[torch.Tensor] = []

            def transform(*, image: np.ndarray, mask: np.ndarray) -> dict:
                flipped = np.flip(mask, axis=1).copy()
                return {
                    "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
                    "mask": torch.from_numpy(flipped),
                }

            def build_weights(target: torch.Tensor) -> torch.Tensor:
                seen.append(target.clone())
                return torch.ones_like(target, dtype=torch.float32)

            sample = SegmentationPatchDataset(
                [record],
                mask_threshold=127,
                transforms=transform,
                segmentation_mode="multiclass",
                target_weight_builder=build_weights,
            )[0]

        self.assertEqual(int(sample["mask"][0, 3]), 1)
        self.assertTrue(torch.equal(seen[0], sample["mask"]))
        self.assertEqual(tuple(sample["loss_weight"].shape), (4, 4))


class GeometryAwareLossTests(unittest.TestCase):
    def test_geometry_cross_entropy_matches_weighted_manual_reduction_and_backpropagates(self) -> None:
        logits = torch.tensor(
            [[[[2.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]]],
            requires_grad=True,
        )
        target = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
        weights = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])
        loss_fn = MulticlassGeometryCEDiceLociCLDiceLoss(iterations=1)
        parts = loss_fn.components(logits, target, weights)
        pixel_ce = F.cross_entropy(logits.float(), target, reduction="none")
        expected_ce = (weights * pixel_ce).sum() / weights.sum()
        self.assertTrue(torch.allclose(parts["geometry_aware_ce"], expected_ce))
        loss = loss_fn(logits, target, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_factory_and_diagnostics_support_geometry_loss(self) -> None:
        config = {
            "name": "multiclass_geometry_ce_dice_loci_cldice",
            "geometry_aware_ce_weight": 0.25,
            "dice_weight": 0.55,
            "soft_cldice_weight": 0.20,
            "iterations": 1,
        }
        loss = build_loss(config)
        logits = torch.randn(1, 3, 8, 8)
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        weights = torch.ones((1, 8, 8))
        metrics = loss_component_metrics(logits, target, config, weights)
        self.assertIsInstance(loss, MulticlassGeometryCEDiceLociCLDiceLoss)
        self.assertEqual(
            set(metrics),
            {
                "geometry_aware_cross_entropy",
                "multiclass_dice_loss",
                "loci_soft_cldice_loss",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


    def test_trainer_consumes_geometry_weight_batch(self) -> None:
        model = torch.nn.Conv2d(3, 3, kernel_size=1)
        loss_config = {
            "name": "multiclass_geometry_ce_dice_loci_cldice",
            "iterations": 1,
        }
        trainer = Trainer(
            model=model,
            loss_fn=build_loss(loss_config),
            optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
            scheduler=None,
            device=torch.device("cpu"),
            train_config={
                "monitor": "val_dice_per_patch",
                "use_tqdm": False,
                "mixed_precision": False,
            },
            loss_config=loss_config,
            logger=None,
            fold_dir=Path("unused"),
            data_config={},
            segmentation_config={
                "mode": "multiclass",
                "classes": {"background": 0, "loci": 1, "inoculum": 2},
            },
        )
        batch = {
            "image": torch.randn(2, 3, 8, 8),
            "mask": torch.zeros((2, 8, 8), dtype=torch.long),
            "loss_weight": torch.ones((2, 8, 8)),
        }

        metrics = trainer._run_epoch(
            [batch], training=False, epoch=1, epochs=1, stage_name="val"
        )

        self.assertTrue(np.isfinite(metrics["val_loss"]))
        self.assertIn("val_geometry_aware_cross_entropy", metrics)


if __name__ == "__main__":
    unittest.main()
