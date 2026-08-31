from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.analyze_soft_skeleton_iterations import (
    required_soft_skeleton_iterations,
    run_analysis,
    sweep_soft_skeleton,
)
from src.metrics.segmentation import soft_skeletonize


class SoftSkeletonIterationAnalysisTests(unittest.TestCase):
    def test_tiled_sweep_matches_production_operator(self) -> None:
        mask = np.zeros((37, 43), dtype=bool)
        mask[3:34, 7:36] = True
        mask[15:22, 1:42] = True

        totals, snapshots, _, kernel_seconds = sweep_soft_skeleton(
            mask,
            minimum=2,
            maximum=6,
            visual_iterations=range(2, 7),
            tile_size=13,
            device=torch.device("cpu"),
        )

        tensor = torch.from_numpy(mask)[None, None]
        for iteration in range(2, 7):
            expected = soft_skeletonize(tensor, iteration)[0, 0].numpy() > 0.5
            np.testing.assert_array_equal(snapshots[iteration], expected)
            self.assertEqual(totals[iteration]["skeleton_pixels"], int(expected.sum()))
            self.assertGreater(kernel_seconds[iteration], 0.0)
        self.assertLess(kernel_seconds[2], kernel_seconds[6])

    def test_required_iterations_tracks_city_block_radius(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        mask[2:9, 2:9] = True
        self.assertEqual(required_soft_skeleton_iterations(mask), 3)
        self.assertEqual(
            required_soft_skeleton_iterations(np.zeros((4, 5), dtype=bool)), 0
        )

    def test_analysis_writes_metrics_summary_and_visual(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask_dir = root / "masks"
            output_dir = root / "results"
            mask_dir.mkdir()
            mask = np.zeros((32, 40), dtype=np.uint8)
            mask[4:28, 12:28] = 255
            Image.fromarray(mask).save(mask_dir / "sample.png")
            Image.fromarray(np.zeros_like(mask)).save(
                mask_dir / "not-selected.png"
            )

            summary = run_analysis(
                mask_dir,
                output_dir,
                minimum=2,
                maximum=10,
                iteration_step=4,
                visual_iterations=[2, 6, 10],
                mask_names=["sample.png"],
                save_full_resolution=True,
                device=torch.device("cpu"),
                tile_size=12,
                preview_size=64,
            )

            self.assertEqual(summary["mask_count"], 1)
            self.assertEqual(summary["tested_iterations"], [2, 6, 10])
            self.assertTrue(summary["all_images_complete_at_max_tested"])
            self.assertTrue((output_dir / "per_image_iterations.csv").is_file())
            self.assertTrue((output_dir / "aggregate_iterations.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "visuals" / "sample.png").is_file())
            full_resolution = output_dir / "full_resolution" / "sample"
            self.assertTrue((full_resolution / "ground_truth.png").is_file())
            for iteration in (2, 6, 10):
                self.assertTrue(
                    (full_resolution / f"soft_skeleton_{iteration}.png").is_file()
                )
            self.assertFalse(
                (output_dir / "full_resolution" / "not-selected").exists()
            )


if __name__ == "__main__":
    unittest.main()
