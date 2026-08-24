from __future__ import annotations

import csv
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data.dataset import SegmentationPatchDataset, compose_multiclass_mask
from src.data.discovery import discover_image_mask_sets
from src.inference import predict_probabilities_on_image
from src.losses.combined import MulticlassCEDiceLociCLDiceLoss
from src.metrics.segmentation import join_region_metrics_from_masks, multiclass_metrics_from_masks
from src.models.factory import build_model
from src.patching import (
    OriginalImageRecord,
    PatchRecord,
    build_original_image_records,
    build_patch_records,
)
from src.inference.test_evaluation import (
    _JOIN_MASK_BOUNDARY_COLOR,
    _MULTICLASS_OVERLAY_COLORS,
    create_test_evaluation_overlay,
    resolve_test_records,
    run_test_evaluation,
)
from src.inference.qualitative_evaluation import CheckpointEntry, run_qualitative_evaluation
from src.utils.config import load_config


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

    def test_test_records_skip_multiclass_dimension_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            mask = np.zeros((8, 10), dtype=np.uint8)
            for stem in ["train", "val", "test", "bad"]:
                save_image(root / "images" / f"{stem}.png", image)
                save_image(root / "loci" / f"{stem}.png", mask)
                inoculum = mask if stem != "bad" else np.zeros((7, 10), dtype=np.uint8)
                save_image(root / "inoculum" / f"{stem}.png", inoculum)
            split_path = root / "splits.csv"
            split_path.write_text(
                (
                    "filename,split\ntrain.png,train\nval.png,validation\n"
                    "test.png,test\nbad.png,train\n"
                ),
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "images_dir": str(root / "images"),
                    "mask_dirs": {
                        "loci": str(root / "loci"),
                        "inoculum": str(root / "inoculum"),
                    },
                },
                "segmentation": {"mode": "multiclass"},
                "data": {"image_extensions": [".png"]},
                "split": {"mode": "csv", "csv_path": str(split_path)},
            }

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                records = resolve_test_records(config)

        self.assertEqual([record.source_id for record in records], ["test.png"])
        warning_text = " ".join(str(item.message) for item in caught)
        self.assertIn("bad (inoculum)", warning_text)
        self.assertIn("bad.png", warning_text)

    def test_optional_join_masks_do_not_make_sources_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((6, 7, 3), dtype=np.uint8)
            mask = np.zeros((6, 7), dtype=np.uint8)
            for stem in ("with_join", "without_join"):
                save_image(root / "images" / f"{stem}.png", image)
                save_image(root / "loci" / f"{stem}.png", mask)
                save_image(root / "inoculum" / f"{stem}.png", mask)
            save_image(root / "join" / "with_join.png", mask)

            sets, diagnostics = discover_image_mask_sets(
                root / "images",
                {"loci": root / "loci", "inoculum": root / "inoculum"},
                [".png"],
                optional_mask_dirs={"join": root / "join"},
            )

        self.assertEqual([image_path.stem for image_path, _ in sets], ["with_join", "without_join"])
        self.assertIn("join", sets[0][1])
        self.assertNotIn("join", sets[1][1])
        self.assertEqual(diagnostics["optional_dimension_mismatches"], [])

    def test_test_records_load_join_masks_for_evaluation_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            mask = np.zeros((4, 5), dtype=np.uint8)
            for stem in ("train", "validation", "test"):
                save_image(root / "images" / f"{stem}.png", image)
                save_image(root / "loci" / f"{stem}.png", mask)
                save_image(root / "inoculum" / f"{stem}.png", mask)
            save_image(root / "join" / "test.png", mask)
            split_path = root / "splits.csv"
            split_path.write_text(
                "filename,split\n"
                "train.png,train\n"
                "validation.png,validation\n"
                "test.png,test\n",
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "images_dir": str(root / "images"),
                    "mask_dirs": {
                        "loci": str(root / "loci"),
                        "inoculum": str(root / "inoculum"),
                    },
                },
                "segmentation": {"mode": "multiclass"},
                "data": {"image_extensions": [".png"]},
                "split": {"mode": "csv", "csv_path": str(split_path)},
                "join_masks": {
                    "enabled": False,
                    "masks_dir": str(root / "join"),
                    "merge_with_loci": False,
                    "evaluation_enabled": True,
                },
            }

            records = resolve_test_records(config)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].mask_paths["join"], root / "join" / "test.png")

    def test_join_masks_merge_into_loci_and_empty_metrics_are_undefined(self) -> None:
        loci = np.zeros((2, 3), dtype=np.uint8)
        inoculum = np.zeros((2, 3), dtype=np.uint8)
        inoculum[0, 0] = 255
        join = np.zeros((2, 3), dtype=np.uint8)
        join[0, :2] = 255
        target, _ = compose_multiclass_mask(
            loci, inoculum, join_mask=join, merge_join_masks=True
        )
        np.testing.assert_array_equal(
            target, np.array([[2, 1, 0], [0, 0, 0]], dtype=np.uint8)
        )
        prediction = torch.tensor([[2, 0, 0], [0, 0, 0]])
        metrics = join_region_metrics_from_masks(
            prediction, torch.from_numpy(target), torch.from_numpy(join > 127)
        )
        self.assertEqual(metrics["join_pixels"], 1)
        self.assertAlmostEqual(metrics["dice_join"], 0.0, places=5)
        self.assertAlmostEqual(metrics["iou_join"], 0.0, places=5)
        empty = join_region_metrics_from_masks(prediction, torch.from_numpy(target), None)
        self.assertEqual(empty, {"join_pixels": 0, "dice_join": None, "iou_join": None})

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

    def test_join_only_patch_is_kept_and_merged_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            join_path = root / "join.png"
            save_image(image_path, np.zeros((4, 4, 3), dtype=np.uint8))
            save_image(loci_path, np.zeros((4, 4), dtype=np.uint8))
            save_image(inoculum_path, np.zeros((4, 4), dtype=np.uint8))
            join = np.zeros((4, 4), dtype=np.uint8)
            join[1:3, 1:3] = 255
            save_image(join_path, join)
            originals = build_original_image_records([(image_path, {
                "loci": loci_path, "inoculum": inoculum_path, "join": join_path,
            })])
            records = build_patch_records(originals, {
                "patch_size": 4, "stride": 4, "filter_empty_patches": True,
                "mask_threshold": 127, "min_foreground_pixels": 1,
                "include_join_masks": True,
            })
            sample = SegmentationPatchDataset(
                records,
                mask_threshold=127,
                segmentation_mode="multiclass",
                merge_join_masks=True,
            )[0]

        self.assertEqual(len(records), 1)
        self.assertEqual(int((sample["mask"] == 1).sum().item()), 4)

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
            join_path = root / "join.png"
            image = np.zeros((6, 7, 3), dtype=np.uint8)
            loci = np.zeros((6, 7), dtype=np.uint8)
            inoculum = np.zeros((6, 7), dtype=np.uint8)
            loci[2:, :3] = 255
            inoculum[:2, 4:] = 255
            join = np.zeros((6, 7), dtype=np.uint8)
            join[4:, 5:] = 255
            save_image(image_path, image)
            save_image(loci_path, loci)
            save_image(inoculum_path, inoculum)
            save_image(join_path, join)
            record = OriginalImageRecord(
                "test.png", image_path, loci_path, 7, 6,
                {"loci": loci_path, "inoculum": inoculum_path, "join": join_path},
            )
            target, _ = compose_multiclass_mask(
                loci, inoculum, join_mask=join, merge_join_masks=True
            )
            probabilities = np.eye(3, dtype=np.float32)[target].transpose(2, 0, 1)
            config = {
                "segmentation": {"mode": "multiclass"},
                "patching": {"mask_threshold": 127},
                "inference": {"decision": "argmax", "save_probabilities": True},
                "join_masks": {"enabled": True, "masks_dir": str(root), "merge_with_loci": True},
                "test_evaluation": {"threshold_sweep": False},
                "loss": {"cldice_smooth": 1.0},
            }
            with patch("src.inference.test_evaluation.resolve_test_records", return_value=[record]):
                result = run_test_evaluation(
                    root / "best.pt", config, root / "evaluation", torch.device("cpu"),
                    model=torch.nn.Identity(),
                    predictor=lambda *_: probabilities,
                )
            self.assertAlmostEqual(result["mean_dice"], 1.0)
            self.assertAlmostEqual(result["mean_dice_join"], 1.0)
            self.assertAlmostEqual(result["mean_iou_join"], 1.0)
            self.assertEqual(result["num_join_images"], 1)
            self.assertEqual(result["threshold"], "argmax")
            for relative in (
                "masks/test_mask.png", "overlays/test_overlay.png",
                "probabilities/test_prob_loci.png",
                "probabilities/test_prob_inoculum.png",
                "test_metrics.csv", "threshold_metrics.csv",
                "multiclass_metrics.png", "summary.json",
            ):
                self.assertTrue((root / "evaluation" / relative).is_file(), relative)

    def test_evaluation_only_join_masks_do_not_change_ordinary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "test.png"
            loci_path = root / "loci.png"
            inoculum_path = root / "inoculum.png"
            join_path = root / "join.png"
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            empty_mask = np.zeros((4, 4), dtype=np.uint8)
            join = np.zeros((4, 4), dtype=np.uint8)
            join[1, 1] = 255
            save_image(image_path, image)
            save_image(loci_path, empty_mask)
            save_image(inoculum_path, empty_mask)
            save_image(join_path, join)
            record = OriginalImageRecord(
                "test.png",
                image_path,
                loci_path,
                4,
                4,
                {"loci": loci_path, "inoculum": inoculum_path, "join": join_path},
            )
            prediction = np.zeros((4, 4), dtype=np.uint8)
            prediction[1, 1] = 1
            probabilities = np.eye(3, dtype=np.float32)[prediction].transpose(2, 0, 1)
            config = {
                "segmentation": {"mode": "multiclass"},
                "patching": {"mask_threshold": 127},
                "inference": {"decision": "argmax", "save_probabilities": False},
                "join_masks": {
                    "enabled": False,
                    "masks_dir": str(root),
                    "merge_with_loci": False,
                    "evaluation_enabled": True,
                },
                "test_evaluation": {"threshold_sweep": False},
                "loss": {"cldice_smooth": 1.0},
            }

            with patch(
                "src.inference.test_evaluation.resolve_test_records",
                return_value=[record],
            ):
                result = run_test_evaluation(
                    root / "best.pt",
                    config,
                    root / "evaluation",
                    torch.device("cpu"),
                    model=torch.nn.Identity(),
                    predictor=lambda *_: probabilities,
                )

        self.assertLess(result["mean_dice_loci"], 1.0e-5)
        self.assertEqual(result["mean_join_pixels"], 1.0)
        self.assertAlmostEqual(result["mean_dice_join"], 1.0)
        self.assertAlmostEqual(result["mean_iou_join"], 1.0)
        self.assertEqual(result["num_join_images"], 1)


    def test_multiclass_overlay_marks_correct_and_wrong_class_overlap(self) -> None:
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        target = np.array([[1, 2, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        prediction = np.array([[1, 2, 2, 2], [0, 0, 0, 0]], dtype=np.uint8)

        overlay = create_test_evaluation_overlay(
            image, target, prediction, multiclass=True, include_legend=False
        )

        scale = 0.65
        np.testing.assert_array_equal(
            overlay[0, 0],
            (_MULTICLASS_OVERLAY_COLORS["Loci correct overlap"] * scale).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            overlay[0, 1],
            (_MULTICLASS_OVERLAY_COLORS["Inoculum correct overlap"] * scale).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            overlay[0, 2],
            (_MULTICLASS_OVERLAY_COLORS["Wrong-class overlap"] * scale).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            overlay[0, 3],
            (_MULTICLASS_OVERLAY_COLORS["Inoculum prediction only"] * scale).astype(np.uint8),
        )

    def test_multiclass_overlay_marks_join_mask_boundary_in_red(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        target = np.ones((5, 5), dtype=np.uint8)
        join = np.ones((5, 5), dtype=np.uint8)

        overlay = create_test_evaluation_overlay(
            image,
            target,
            target,
            multiclass=True,
            join_mask=join,
            include_legend=False,
        )

        scale = 0.65
        correct_color = (
            _MULTICLASS_OVERLAY_COLORS["Loci correct overlap"] * scale
        ).astype(np.uint8)
        expected_boundary = (
            (1.0 - scale) * correct_color
            + scale * _JOIN_MASK_BOUNDARY_COLOR
        ).astype(np.uint8)
        np.testing.assert_array_equal(overlay[0, 2], expected_boundary)
        np.testing.assert_array_equal(overlay[2, 2], correct_color)

    def test_multiclass_config_uses_resnet50_encoder_and_join_masks(self) -> None:
        config = load_config("multiclass-config.yaml")

        self.assertEqual(config["model"]["name"], "unetplusplus_resnet50")
        self.assertEqual(config["model"]["encoder_name"], "resnet50")
        self.assertEqual(
            config["join_masks"],
            {
                "enabled": False,
                "masks_dir": "data/join_masks",
                "merge_with_loci": False,
                "evaluation_enabled": True,
            },
        )

        config["model"]["encoder_weights"] = None
        model = build_model(config["model"]).eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 64))

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
                    "crop_patch_grid": [1, 1], "min_foreground_ratio": 0.1,
                    "max_foreground_ratio": 0.3, "selection_seed": 1,
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
                patch("src.inference.qualitative_evaluation.load_config", return_value=config),
                patch(
                    "src.inference.qualitative_evaluation.resolve_qualitative_pairs",
                    return_value=(
                        [(image_path, {"loci": loci_path, "inoculum": inoculum_path})],
                        {"missing_masks": {}, "missing_images": {}},
                        "test",
                    ),
                ),
                patch("src.inference.qualitative_evaluation.discover_manifest_checkpoints", return_value=[entry]),
                patch("src.inference.qualitative_evaluation.build_model", return_value=torch.nn.Identity()),
                patch("src.inference.qualitative_evaluation.load_checkpoint"),
                patch("src.inference.qualitative_evaluation.predict_crop_probabilities", return_value=probabilities),
            ):
                result = run_qualitative_evaluation(
                    root, output_dir=output, crop_patch_grid=(1, 1),
                    min_foreground_ratio=0.1, max_foreground_ratio=0.3,
                    device_name="cpu",
                )
            self.assertFalse(result["skipped"])
            crop = result["selected_crops"][image_path.stem]
            self.assertEqual(crop.selection_reason, "in_range")
            self.assertAlmostEqual(crop.foreground_ratio, 14.0 / 64.0)
            self.assertTrue((output / "grids" / "test.png").is_file())
            self.assertTrue((output / "eval_metrics.csv").is_file())
            self.assertTrue((output / "selected_crops.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue(any((output / "masks").glob("*_mask.png")))
            self.assertEqual(len(list((output / "probabilities").glob("*.png"))), 2)

if __name__ == "__main__":
    unittest.main()
