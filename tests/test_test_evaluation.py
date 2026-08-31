from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.test_evaluation import (
    _BINARY_OVERLAY_COLORS,
    create_test_evaluation_overlay,
    default_config_path,
    run_test_evaluation,
    threshold_values,
)


class TestEvaluationTests(unittest.TestCase):
    def _make_config(self, root: Path) -> dict:
        images_dir = root / "images"
        masks_dir = root / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        for name, foreground in (("train", False), ("val", False), ("test", True)):
            image = np.full((4, 5, 3), 120, dtype=np.uint8)
            mask = np.zeros((4, 5), dtype=np.uint8)
            if foreground:
                mask[:2, :2] = 255
            Image.fromarray(image).save(images_dir / f"{name}.png")
            Image.fromarray(mask).save(masks_dir / f"{name}.png")
        (root / "splits.csv").write_text(
            "filename,split\ntrain.png,train\nval.png,validation\ntest.png,test\n",
            encoding="utf-8",
        )
        return {
            "paths": {"images_dir": str(images_dir), "mask_dirs": {"loci": str(masks_dir)}},
            "segmentation": {"target": "loci"},
            "data": {"image_extensions": [".png"]},
            "split": {"mode": "csv", "csv_path": str(root / "splits.csv")},
            "patching": {"patch_size": 4, "stride": 4, "mask_threshold": 127},
            "inference": {"threshold": 0.5},
            "test_evaluation": {"threshold_start": 0.5, "threshold_stop": 1.0, "threshold_step": 0.01},
        }

    def test_writes_test_artifacts_and_threshold_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._make_config(root)
            config["split"]["mode"] = "csv_kfold"
            output_dir = root / "test-evaluation"

            def predictor(*_args):
                probabilities = np.zeros((4, 5), dtype=np.float32)
                probabilities[:2, :2] = 0.8
                return probabilities

            result = run_test_evaluation(
                root / "best.pt",
                config,
                output_dir,
                torch.device("cpu"),
                model=torch.nn.Identity(),
                predictor=predictor,
            )

            self.assertEqual(result["num_test_images"], 1)
            self.assertAlmostEqual(result["mean_dice"], 1.0)
            self.assertTrue((output_dir / "masks" / "test_mask.png").is_file())
            self.assertTrue((output_dir / "overlays" / "test_overlay.png").is_file())
            for metric_name in (
                "dice",
                "iou",
                "precision",
                "recall",
                "cldice",
                "predicted_foreground_fraction",
            ):
                self.assertTrue((output_dir / f"{metric_name}_by_threshold.png").is_file())
            with (output_dir / "test_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            with (output_dir / "threshold_metrics.csv").open(newline="", encoding="utf-8") as handle:
                threshold_rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_id"] for row in metric_rows], ["test.png", "mean"])
            self.assertAlmostEqual(float(metric_rows[0]["precision"]), 1.0)
            self.assertAlmostEqual(float(metric_rows[0]["recall"]), 1.0)
            self.assertAlmostEqual(float(metric_rows[0]["cldice"]), 1.0)
            self.assertAlmostEqual(float(metric_rows[0]["predicted_foreground_fraction"]), 0.2)
            self.assertAlmostEqual(float(metric_rows[1]["dice"]), 1.0)
            self.assertEqual(len(threshold_rows), 51)
            self.assertEqual(threshold_rows[0]["threshold"], "0.5")
            self.assertEqual(threshold_rows[-1]["threshold"], "1.0")

    def test_test_evaluation_uses_half_patch_stride_without_mutating_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._make_config(root)
            observed_strides = []

            def predictor(_model, _image_path, evaluation_config, _device):
                observed_strides.append(evaluation_config["patching"]["stride"])
                return np.zeros((4, 5), dtype=np.float32)

            run_test_evaluation(
                root / "best.pt",
                config,
                root / "test-evaluation",
                torch.device("cpu"),
                model=torch.nn.Identity(),
                predictor=predictor,
            )

            self.assertEqual(observed_strides, [2])
            self.assertEqual(config["patching"]["stride"], 4)

    def test_binary_overlay_distinguishes_ground_truth_prediction_and_overlap(self) -> None:
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        target = np.array([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
        prediction = np.array([[0, 0, 255, 255], [0, 0, 0, 0]], dtype=np.uint8)

        overlay = create_test_evaluation_overlay(
            image, target, prediction, multiclass=False, include_legend=False
        )

        expected_scale = 0.65
        np.testing.assert_array_equal(overlay[0, 0], image[0, 0])
        np.testing.assert_array_equal(
            overlay[0, 1],
            (_BINARY_OVERLAY_COLORS["Ground truth only"] * expected_scale).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            overlay[0, 2],
            (_BINARY_OVERLAY_COLORS["Prediction only"] * expected_scale).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            overlay[0, 3],
            (_BINARY_OVERLAY_COLORS["Correct overlap"] * expected_scale).astype(np.uint8),
        )

    def test_binary_overlay_includes_bottom_right_legend(self) -> None:
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        mask = np.zeros((120, 240), dtype=np.uint8)

        overlay = create_test_evaluation_overlay(image, mask, mask, multiclass=False)

        self.assertTrue(np.any(overlay[-50:, -180:] != 0))

    def test_skips_incomplete_and_absent_future_csv_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._make_config(root)
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
                root / "images" / "awaiting_mask.png"
            )
            with (root / "splits.csv").open("a", encoding="utf-8") as handle:
                handle.write("awaiting_mask.png,train\nfuture_image.png,test\n")

            with self.assertWarnsRegex(RuntimeWarning, "awaiting_mask"):
                result = run_test_evaluation(
                    root / "best.pt",
                    config,
                    root / "test-evaluation",
                    torch.device("cpu"),
                    model=torch.nn.Identity(),
                    predictor=lambda *_: np.zeros((4, 5), dtype=np.float32),
                )

        self.assertEqual(result["num_test_images"], 1)

    def test_threshold_defaults_include_both_endpoints(self) -> None:
        values = threshold_values({"test_evaluation": {"threshold_start": 0.5, "threshold_stop": 1.0, "threshold_step": 0.01}})
        self.assertEqual(len(values), 51)
        self.assertEqual(values[0], 0.5)
        self.assertEqual(values[-1], 1.0)

    def test_default_config_path_uses_checkpoint_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            checkpoint = run_dir / "fold_0" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            config_path = run_dir / "config.yaml"
            config_path.write_text("project: {}\n", encoding="utf-8")
            self.assertEqual(default_config_path(checkpoint), config_path)


if __name__ == "__main__":
    unittest.main()
