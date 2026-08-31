from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.benchmark_cldice_patches import (
    benchmark_patch_device,
    run_patch_benchmark,
    select_foreground_patches,
)


class ClDicePatchBenchmarkTests(unittest.TestCase):
    def test_foreground_patch_selection_is_seeded_and_nonempty(self) -> None:
        mask = np.zeros((64, 80), dtype=bool)
        mask[8:56, 12:68] = True

        first_prediction, first_target, first_records = select_foreground_patches(
            mask,
            mask,
            patch_size=16,
            stride=8,
            num_patches=10,
            seed=42,
        )
        second_prediction, second_target, second_records = select_foreground_patches(
            mask,
            mask,
            patch_size=16,
            stride=8,
            num_patches=10,
            seed=42,
        )

        self.assertEqual(first_prediction.shape, (10, 16, 16))
        np.testing.assert_array_equal(first_prediction, first_target)
        np.testing.assert_array_equal(first_prediction, second_prediction)
        np.testing.assert_array_equal(first_target, second_target)
        self.assertEqual(first_records, second_records)
        self.assertTrue(all(record["foreground_pixels"] > 0 for record in first_records))

    def test_cpu_patch_benchmark_honors_batching(self) -> None:
        patches = np.zeros((5, 24, 24), dtype=bool)
        patches[:, 4:20, 10:14] = True

        result, prediction_skeletons, target_skeletons = benchmark_patch_device(
            patches,
            patches,
            device=torch.device("cpu"),
            batch_size=3,
            repeats=1,
            smooth=1e-6,
        )

        self.assertEqual(result["num_patches"], 5)
        self.assertEqual(result["batch_size"], 3)
        self.assertAlmostEqual(result["cldice"], 1.0)
        self.assertGreater(result["median_patches_per_second"], 0.0)
        np.testing.assert_array_equal(prediction_skeletons, target_skeletons)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_full_patch_benchmark_writes_equivalent_cuda_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask_path = root / "mask.png"
            output_dir = root / "results"
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[8:56, 20:44] = 255
            Image.fromarray(mask).save(mask_path)

            summary = run_patch_benchmark(
                mask_path,
                mask_path,
                output_dir,
                patch_size=16,
                stride=8,
                num_patches=4,
                batch_size=3,
                repeats=1,
            )

            self.assertEqual(summary["actual_num_patches"], 4)
            self.assertEqual(summary["max_cldice_absolute_difference"], 0.0)
            self.assertEqual(
                summary["prediction_skeleton_total_different_pixels"], 0
            )
            self.assertEqual(summary["target_skeleton_total_different_pixels"], 0)
            self.assertTrue((output_dir / "summary.json").is_file())
            for artifact in summary["artifacts"].values():
                self.assertTrue((output_dir / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
