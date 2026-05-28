from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from src.data.sampling import build_balanced_resolution_source_sampler
from src.patching import (
    OriginalImageRecord,
    PatchRecord,
    _compute_positions,
    crop_scaled_mask_patch,
    resolve_scale_specs,
)


MULTISCALE_CONFIG = {
    "enabled": True,
    "include_native": True,
    "target_long_edges": [1200, 1600, 2400, 3200],
    "max_scale": 1.0,
    "deduplicate_scale_tolerance": 0.03,
}


class MultiScaleTests(unittest.TestCase):
    def test_scale_generation_for_large_and_small_images(self) -> None:
        large = OriginalImageRecord("large", Path("image.tif"), Path("mask.png"), 9607, 6820)
        small = OriginalImageRecord("small", Path("image.tif"), Path("mask.png"), 1600, 1200)

        large_specs = resolve_scale_specs(large, MULTISCALE_CONFIG)
        small_specs = resolve_scale_specs(small, MULTISCALE_CONFIG)

        self.assertEqual(
            [spec["scale_label"] for spec in large_specs],
            ["long_edge_1200", "long_edge_1600", "long_edge_2400", "long_edge_3200", "native"],
        )
        self.assertEqual(
            [(spec["scaled_width"], spec["scaled_height"]) for spec in large_specs],
            [(1200, 852), (1600, 1136), (2400, 1704), (3200, 2272), (9607, 6820)],
        )
        self.assertEqual(
            [spec["scale_label"] for spec in small_specs],
            ["long_edge_1200", "native"],
        )

    def test_virtual_patch_geometry_and_foreground_preservation(self) -> None:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[10:1014, 512] = 255

        patch = crop_scaled_mask_patch(
            mask,
            x=0,
            y=0,
            patch_size=256,
            scale=0.25,
            mask_threshold=127,
            resampling="foreground_preserving",
        )

        self.assertEqual(patch.shape, (256, 256))
        self.assertGreater(int((patch > 127).sum()), 0)
        self.assertEqual(_compute_positions(1200, 256, 128)[-1], 944)

    def test_balanced_sampler_uses_native_patch_count_epoch_length(self) -> None:
        records = [
            PatchRecord("a", Path("i"), Path("m"), 0, 0, 256, 1.0, 1600, 1200, "bucket_1600", "native"),
            PatchRecord("a", Path("i"), Path("m"), 1, 0, 256, 1.0, 1600, 1200, "bucket_1600", "native"),
            PatchRecord("b", Path("i"), Path("m"), 0, 0, 256, 0.5, 1200, 900, "bucket_1200", "long_edge_1200"),
            PatchRecord("b", Path("i"), Path("m"), 1, 0, 256, 0.5, 1200, 900, "bucket_1200", "long_edge_1200"),
            PatchRecord("b", Path("i"), Path("m"), 2, 0, 256, 0.5, 1200, 900, "bucket_1200", "long_edge_1200"),
            PatchRecord("c", Path("i"), Path("m"), 0, 0, 256, 0.5, 1200, 900, "bucket_1200", "long_edge_1200"),
        ]

        _, diagnostics = build_balanced_resolution_source_sampler(
            records,
            {
                "strategy": "balanced_resolution_source",
                "samples_per_epoch": "native_patch_count",
                "replacement": True,
            },
            generator=torch.Generator().manual_seed(42),
        )

        self.assertEqual(diagnostics["samples_per_epoch"], 2)
        self.assertAlmostEqual(
            diagnostics["effective_samples_per_bucket"]["bucket_1200"],
            diagnostics["effective_samples_per_bucket"]["bucket_1600"],
        )
        self.assertAlmostEqual(
            diagnostics["weight_by_resolution_bucket_source"]["bucket_1200::b"],
            diagnostics["weight_by_resolution_bucket_source"]["bucket_1200::c"],
        )


if __name__ == "__main__":
    unittest.main()
