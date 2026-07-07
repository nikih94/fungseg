from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml
from PIL import Image

from src.patching import (
    OriginalImageRecord,
    _compute_positions,
    build_patch_records,
    compute_shifted_positions,
    crop_and_pad_array,
    crop_scaled_mask_patch,
)


PATCHING_CONFIG = {
    "patch_size": 256,
    "overlap": 128,
    "stride": 128,
    "filter_empty_patches": False,
    "mask_threshold": 127,
    "min_foreground_pixels": 1,
    "image_resampling": "lanczos",
    "mask_resampling": "foreground_preserving",
    "train": {
        "random_offset": {"enabled": True, "max_fraction_of_patch": 0.5},
        "scaled_context": {
            "enabled": True,
            "probability": 0.25,
            "max_scale": 2.0,
            "beta_alpha": 1.0,
            "beta_beta": 4.0,
        },
    },
    "validation": {
        "random_offset": {"enabled": False},
        "scaled_context": {"enabled": False},
    },
}


def _write_pair(root: Path, width: int = 1024, height: int = 1024) -> OriginalImageRecord:
    image_path = root / "image.tif"
    mask_path = root / "mask.png"
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 128
    mask = np.ones((height, width), dtype=np.uint8) * 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return OriginalImageRecord(image_path.name, image_path, mask_path, width, height)


class DynamicPatchingTests(unittest.TestCase):
    def test_shifted_positions_keep_edges(self) -> None:
        self.assertEqual(_compute_positions(1200, 256, 128)[-1], 944)
        self.assertEqual(compute_shifted_positions(512, 256, 128, 37), [0, 37, 165, 256])

    def test_epoch_plans_are_reproducible_and_change_by_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = _write_pair(Path(tmp))
            first = build_patch_records([record], PATCHING_CONFIG, phase="train", epoch=3, base_seed=42)
            second = build_patch_records([record], PATCHING_CONFIG, phase="train", epoch=3, base_seed=42)
            different = build_patch_records([record], PATCHING_CONFIG, phase="train", epoch=4, base_seed=42)

        self.assertEqual([(item.x, item.y, item.scale) for item in first], [(item.x, item.y, item.scale) for item in second])
        self.assertNotEqual([(item.x, item.y, item.scale) for item in first], [(item.x, item.y, item.scale) for item in different])

    def test_scaled_context_distribution_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = _write_pair(Path(tmp), width=4096, height=4096)
            records = build_patch_records([record], PATCHING_CONFIG, phase="train", epoch=1, base_seed=42)

        scales = [item.scale for item in records]
        scaled = [scale for scale in scales if scale > 1.0]
        self.assertTrue(all(1.0 <= scale <= 2.0 for scale in scales))
        self.assertGreater(len(scaled) / len(records), 0.15)
        self.assertLess(len(scaled) / len(records), 0.30)
        self.assertLess(sum(scale > 1.5 for scale in scales) / len(records), 0.05)

    def test_border_patches_fall_back_when_context_does_not_fit(self) -> None:
        config = {
            **PATCHING_CONFIG,
            "train": {
                "random_offset": {"enabled": False},
                "scaled_context": {
                    "enabled": True,
                    "probability": 1.0,
                    "max_scale": 2.0,
                    "beta_alpha": 1.0,
                    "beta_beta": 4.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            record = _write_pair(Path(tmp), width=512, height=512)
            records = build_patch_records([record], config, phase="train", epoch=1, base_seed=42)

        by_xy = {(item.x, item.y): item for item in records}
        self.assertEqual(by_xy[(0, 0)].scale, 1.0)
        self.assertGreater(by_xy[(128, 128)].scale, 1.0)
        self.assertEqual(by_xy[(256, 256)].scale, 1.0)

    def test_scaled_crop_resizes_context_and_preserves_foreground(self) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:412, 256] = 255
        normal = crop_scaled_mask_patch(mask, 128, 128, 256, 1.0, 127)
        expected = crop_and_pad_array(mask, 128, 128, 256)
        scaled = crop_scaled_mask_patch(mask, 128, 128, 256, 2.0, 127)

        self.assertTrue(np.array_equal(normal, expected))
        self.assertEqual(scaled.shape, (256, 256))
        self.assertGreater(int((scaled > 127).sum()), 0)

    def test_explain_cli_runs_on_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            masks_dir = root / "masks"
            images_dir.mkdir()
            masks_dir.mkdir()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.ones((512, 512), dtype=np.uint8) * 255
            Image.fromarray(image).save(images_dir / "sample.tif")
            Image.fromarray(mask).save(masks_dir / "sample.png")
            config_path = root / "config.yaml"
            config = {
                "project": {"name": "test"},
                "paths": {
                    "images_dir": str(images_dir),
                    "masks_dir": str(masks_dir),
                    "outputs_dir": str(root / "outputs"),
                },
                "data": {"image_extensions": [".tif"], "num_workers": 0, "batch_size": 1},
                "patching": PATCHING_CONFIG,
                "train": {"seed": 7},
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "src.patching.explain", "--config", str(config_path), "--epoch", "1"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("Images matched: 1", result.stdout)
        self.assertIn("Scaled-context patches:", result.stdout)
        self.assertIn("Patches by source and source-crop resolution:", result.stdout)
        self.assertIn("<=256", result.stdout)
        self.assertIn("<=512", result.stdout)
        self.assertIn("percent", result.stdout)
        self.assertIn("100.0%", result.stdout)


if __name__ == "__main__":
    unittest.main()
