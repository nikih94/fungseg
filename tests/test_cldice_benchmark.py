from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.benchmark_cldice import (
    benchmark_device,
    create_overlap_image,
    load_binary_mask,
    run_benchmark,
    skimage_skeletonize_masks,
    skeleton_similarity,
    torch_zhang_skeletonize_masks,
)

from src.metrics.segmentation import cldice_score_from_masks


class ClDiceBenchmarkTests(unittest.TestCase):
    def test_mask_loading_supports_threshold_and_class_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.png"
            Image.fromarray(
                np.array([[0, 1], [128, 255]], dtype=np.uint8)
            ).save(path)

            thresholded = load_binary_mask(
                path, threshold=127, foreground_value=None
            )
            selected = load_binary_mask(path, threshold=127, foreground_value=1)

        np.testing.assert_array_equal(
            thresholded, np.array([[False, False], [True, True]])
        )
        np.testing.assert_array_equal(
            selected, np.array([[False, True], [False, False]])
        )

    def test_skeleton_similarity_and_overlap_colors_report_differences(self) -> None:
        first = np.array([[1, 1], [0, 0]], dtype=bool)
        second = np.array([[1, 0], [1, 0]], dtype=bool)

        metrics = skeleton_similarity(first, second)
        overlap = create_overlap_image(first, second)

        self.assertEqual(metrics["intersection_pixels"], 1)
        self.assertEqual(metrics["different_pixels"], 2)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        np.testing.assert_array_equal(overlap[0, 0], (255, 255, 255))
        np.testing.assert_array_equal(overlap[0, 1], (255, 0, 0))
        np.testing.assert_array_equal(overlap[1, 0], (0, 255, 255))
        np.testing.assert_array_equal(overlap[1, 1], (0, 0, 0))

    def test_cpu_benchmark_returns_metric_timing_and_skeletons(self) -> None:
        target = np.zeros((24, 32), dtype=bool)
        prediction = np.zeros_like(target)
        target[4:20, 14:18] = True
        prediction[5:20, 14:18] = True

        result, prediction_skeleton, target_skeleton = benchmark_device(
            prediction,
            target,
            device=torch.device("cpu"),
            repeats=1,
            smooth=1e-6,
        )

        self.assertEqual(result["device"], "cpu")
        self.assertIn("skimage.morphology.skeletonize", result["algorithm"])
        self.assertGreater(result["median_seconds"], 0.0)
        self.assertGreaterEqual(result["cldice"], 0.0)
        self.assertLessEqual(result["cldice"], 1.0)
        self.assertAlmostEqual(
            result["cldice"],
            cldice_score_from_masks(
                torch.from_numpy(prediction), torch.from_numpy(target)
            ),
        )
        self.assertEqual(prediction_skeleton.shape, prediction.shape)
        self.assertEqual(target_skeleton.shape, target.shape)
        self.assertGreater(int(prediction_skeleton.sum()), 0)
        self.assertGreater(int(target_skeleton.sum()), 0)

    def test_torch_zhang_matches_skimage_for_branches_and_loops(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 29:35] = True
        mask[28:36, 8:56] = True
        mask[8:56, 8:14] = True
        mask[8:14, 8:56] = True

        reference = skimage_skeletonize_masks(mask)
        torch_result = torch_zhang_skeletonize_masks(
            torch.from_numpy(mask)
        ).numpy()

        np.testing.assert_array_equal(torch_result, reference)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_full_benchmark_writes_equivalent_cpu_cuda_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prediction_path = root / "prediction.png"
            target_path = root / "target.png"
            output_dir = root / "results"
            target = np.zeros((24, 32), dtype=np.uint8)
            prediction = np.zeros_like(target)
            target[4:20, 14:18] = 255
            prediction[5:20, 14:18] = 255
            Image.fromarray(prediction).save(prediction_path)
            Image.fromarray(target).save(target_path)

            summary = run_benchmark(
                prediction_path,
                target_path,
                output_dir,
                repeats=1,
            )
            saved_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )

            self.assertAlmostEqual(
                summary["cpu"]["cldice"], summary["gpu"]["cldice"], places=7
            )
            self.assertEqual(summary["skeletonizer"], "paper")
            self.assertIn(
                "skimage.morphology.skeletonize",
                summary["cpu"]["algorithm"],
            )
            self.assertEqual(
                summary["gpu"]["algorithm"],
                "PyTorch Zhang-Suen thinning",
            )
            self.assertEqual(
                summary["prediction_skeleton_cpu_gpu"]["different_pixels"], 0
            )
            self.assertEqual(
                summary["target_skeleton_cpu_gpu"]["different_pixels"], 0
            )
            self.assertEqual(saved_summary["speedup_median"], summary["speedup_median"])
            self.assertTrue((output_dir / "timings.csv").is_file())
            for artifact in summary["artifacts"].values():
                self.assertTrue((output_dir / artifact).is_file(), artifact)

if __name__ == "__main__":
    unittest.main()
