from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data.dataset import SegmentationPatchDataset, compose_multiclass_mask
from src.data.discovery import discover_image_mask_sets
from src.inference import predict_probabilities_on_image
from src.losses.combined import MulticlassCEDiceLociCLDiceLoss
from src.metrics.segmentation import multiclass_metrics_from_masks
from src.models.factory import build_model
from src.patching import (
    OriginalImageRecord,
    PatchRecord,
    build_original_image_records,
    build_patch_records,
)
from src.test_evaluation import run_test_evaluation
from src.qualitative_evaluation import CheckpointEntry, run_qualitative_evaluation


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class ConstantMulticlassModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        logits = torch.zeros((batch, 3, height, width), device=images.device)
        logits[:, 1] = 1.0
        logits[:, 2, : height // 2] = 2.0
        return logits


class MulticlassPipelineTests(unittest.TestCase):
    def test_discovers_complete_dimension_matched_triplets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            mask = np.zeros((8, 10), dtype=np.uint8)
            save_image(root / "images" / "ok.png", image)
            save_image(root / "loci" / "ok.png", mask)
            save_image(root / "inoculum" / "ok.png", mask)
            save_image(root / "images" / "missing.png", image)
            save_image(root / "loci" / "missing.png", mask)
            save_image(root / "images" / "bad.png", image)
            save_image(root / "loci" / "bad.png", mask)
            save_image(root / "inoculum" / "bad.png", np.zeros((7, 10), dtype=np.uint8))

            sets, diagnostics = discover_image_mask_sets(
                root / "images",
                {"loci": root / "loci", "inoculum": root / "inoculum"},
                [".png"],
            )
            self.assertEqual([item[0].stem for item in sets], ["ok"])
            self.assertEqual(diagnostics["missing_masks"]["inoculum"], ["missing"])
            self.assertEqual(diagnostics["dimension_mismatches"][0]["class"], "inoculum")

    def test_composition_precedence_and_overlap_diagnostics(self) -> None:
        loci = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        inoculum = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        target, diagnostics = compose_multiclass_mask(loci, inoculum)
        np.testing.assert_array_equal(target, np.array([[2, 1], [2, 0]], dtype=np.uint8))
        self.assertEqual(diagnostics["overlap_pixels"], 1)
        self.assertEqual(diagnostics["overlap_fraction"], 0.25)

    def test_union_filter_keeps_inoculum_only_patch_and_dataset_is_long_hw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            save_image(image_path, np.zeros((8, 8, 3), dtype=np.uint8))
            save_image(loci_path, np.zeros((8, 8), dtype=np.uint8))
            inoculum = np.zeros((8, 8), dtype=np.uint8)
            inoculum[:4, :4] = 255
            save_image(inoculum_path, inoculum)
            originals = build_original_image_records([
                (image_path, {"loci": loci_path, "inoculum": inoculum_path})
            ])
            records = build_patch_records(originals, {
                "patch_size": 8, "stride": 8, "filter_empty_patches": True,
                "mask_threshold": 127, "min_foreground_pixels": 1,
            })
            self.assertEqual(len(records), 1)
            sample = SegmentationPatchDataset(
                records, mask_threshold=127, segmentation_mode="multiclass"
            )[0]
            self.assertEqual(sample["mask"].shape, (8, 8))
            self.assertEqual(sample["mask"].dtype, torch.long)
            self.assertEqual(set(sample["mask"].unique().tolist()), {0, 2})

    def test_unetplusplus_model_outputs_three_logits_per_pixel(self) -> None:
        model = build_model({
            "name": "unetplusplus_resnet34", "encoder_weights": None,
            "in_channels": 3, "num_classes": 3,
            "decoder_channels": [256, 128, 64, 32, 16],
            "decoder_normalization": "batchnorm",
            "decoder_attention_type": None,
        }).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output.shape, (1, 3, 64, 64))

    def test_multiclass_loss_components_are_finite_and_loci_target_drives_cldice(self) -> None:
        torch.manual_seed(4)
        logits = torch.randn(2, 3, 12, 12, requires_grad=True)
        targets = torch.zeros((2, 12, 12), dtype=torch.long)
        targets[:, 3:9, 5:7] = 1
        targets[:, :3, :3] = 2
        loss_fn = MulticlassCEDiceLociCLDiceLoss(iterations=3)
        parts = loss_fn.components(logits, targets)
        for value in parts.values():
            self.assertTrue(torch.isfinite(value))
        loss = loss_fn(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        changed_non_loci = targets.clone()
        changed_non_loci[:, 9:, 9:] = 2
        parts_changed = loss_fn.components(logits.detach(), changed_non_loci)
        self.assertTrue(torch.equal(targets == 1, changed_non_loci == 1))
        self.assertAlmostEqual(
            parts["loci_cldice"].item(), parts_changed["loci_cldice"].item(), places=7
        )

    def test_softmax_stitching_argmax_and_per_class_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            save_image(image_path, np.zeros((9, 11, 3), dtype=np.uint8))
            config = {
                "segmentation": {"mode": "multiclass"},
                "model": {"num_classes": 3},
                "data": {"image_size": None},
                "augmentations": {},
                "patching": {"patch_size": 8, "stride": 4},
            }
            probabilities = predict_probabilities_on_image(
                ConstantMulticlassModel(), image_path, config, torch.device("cpu")
            )
            self.assertEqual(probabilities.shape, (3, 9, 11))
            np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-6)
            prediction = torch.from_numpy(probabilities.argmax(axis=0))
            metrics = multiclass_metrics_from_masks(prediction, prediction)
            self.assertAlmostEqual(metrics["dice_loci"], 1.0)
            self.assertAlmostEqual(metrics["dice_inoculum"], 1.0)
            self.assertAlmostEqual(metrics["dice_macro_foreground"], 1.0)

    def test_multiclass_test_evaluation_writes_argmax_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "test.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            image = np.zeros((6, 7, 3), dtype=np.uint8)
            loci = np.zeros((6, 7), dtype=np.uint8)
            inoculum = np.zeros((6, 7), dtype=np.uint8)
            loci[2:, :3] = 255
            inoculum[:2, 4:] = 255
            save_image(image_path, image)
            save_image(loci_path, loci)
            save_image(inoculum_path, inoculum)
            record = OriginalImageRecord(
                "test.png", image_path, loci_path, 7, 6,
                {"loci": loci_path, "inoculum": inoculum_path},
            )
            target, _ = compose_multiclass_mask(loci, inoculum)
            probabilities = np.eye(3, dtype=np.float32)[target].transpose(2, 0, 1)
            config = {
                "segmentation": {"mode": "multiclass"},
                "patching": {"mask_threshold": 127},
                "inference": {"decision": "argmax", "save_probabilities": True},
                "test_evaluation": {"threshold_sweep": False, "cldice_iterations": 2},
                "loss": {"cldice_smooth": 1.0},
            }
            with patch("src.test_evaluation.resolve_test_records", return_value=[record]):
                result = run_test_evaluation(
                    root / "best.pt", config, root / "evaluation", torch.device("cpu"),
                    model=torch.nn.Identity(),
                    predictor=lambda *_: probabilities,
                )
            self.assertAlmostEqual(result["mean_dice"], 1.0)
            self.assertEqual(result["threshold"], "argmax")
            for relative in (
                "masks/test_mask.png", "overlays/test_overlay.png",
                "probabilities/test_prob_loci.png",
                "probabilities/test_prob_inoculum.png",
                "test_metrics.csv", "threshold_metrics.csv",
                "multiclass_metrics.png", "summary.json",
            ):
                self.assertTrue((root / "evaluation" / relative).is_file(), relative)


    def test_multiclass_qualitative_evaluation_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "test.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            loci = np.zeros((8, 8), dtype=np.uint8)
            inoculum = np.zeros((8, 8), dtype=np.uint8)
            loci[2:6, 2:4] = 255
            inoculum[:2, 5:] = 255
            save_image(image_path, image)
            save_image(loci_path, loci)
            save_image(inoculum_path, inoculum)
            target, _ = compose_multiclass_mask(loci, inoculum)
            probabilities = np.eye(3, dtype=np.float32)[target].transpose(2, 0, 1)
            config = {
                "segmentation": {"mode": "multiclass"},
                "model": {"num_classes": 3},
                "data": {"image_size": None},
                "augmentations": {},
                "patching": {"patch_size": 8, "stride": 8, "mask_threshold": 127},
                "train": {"device": "cpu", "mixed_precision": False},
                "inference": {"decision": "argmax", "save_probabilities": True},
                "qualitative_evaluation": {
                    "crop_patch_grid": [1, 1], "min_foreground_ratio": 0.0,
                    "max_foreground_ratio": 1.0, "selection_seed": 1,
                },
                "split": {"mode": "train_val"},
            }
            entry = CheckpointEntry(
                fold=0, checkpoint="best.pt", path=root / "best.pt",
                reason="global_best", epoch=1, epoch_start=1, epoch_end=1,
                monitor="val_dice_macro_foreground", monitor_value=1.0,
            )
            output = root / "qualitative"
            with (
                patch("src.qualitative_evaluation.load_config", return_value=config),
                patch(
                    "src.qualitative_evaluation.resolve_qualitative_pairs",
                    return_value=(
                        [(image_path, {"loci": loci_path, "inoculum": inoculum_path})],
                        {"missing_masks": {}, "missing_images": {}},
                        "test",
                    ),
                ),
                patch("src.qualitative_evaluation.discover_manifest_checkpoints", return_value=[entry]),
                patch("src.qualitative_evaluation.build_model", return_value=torch.nn.Identity()),
                patch("src.qualitative_evaluation.load_checkpoint"),
                patch("src.qualitative_evaluation.predict_crop_probabilities", return_value=probabilities),
            ):
                result = run_qualitative_evaluation(
                    root, output_dir=output, crop_patch_grid=(1, 1),
                    min_foreground_ratio=0.0, max_foreground_ratio=1.0,
                    device_name="cpu",
                )
            self.assertFalse(result["skipped"])
            self.assertTrue((output / "grids" / "test.png").is_file())
            self.assertTrue((output / "eval_metrics.csv").is_file())
            self.assertTrue((output / "selected_crops.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue(any((output / "masks").glob("*_mask.png")))
            self.assertEqual(len(list((output / "probabilities").glob("*.png"))), 2)

if __name__ == "__main__":
    unittest.main()
