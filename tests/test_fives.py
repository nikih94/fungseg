from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import TensorDataset

from src.data.fives import (
    FivesPatchDataset,
    build_fives_patch_records,
    centered_fives_coordinates,
    discover_fives_pairs,
    load_fives_training_records,
)
from src.train import combine_training_datasets
from src.utils.config import load_config
from src.visualize_fives_patches import create_fives_patch_visualization


def _write_pair(
    root: Path,
    name: str = "sample.png",
    *,
    image_size: tuple[int, int] = (32, 32),
    mask_size: tuple[int, int] | None = None,
) -> tuple[Path, Path]:
    images_dir = root / "Original"
    masks_dir = root / "Ground truth"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 100
    mask_width, mask_height = mask_size or image_size
    mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
    mask[mask_height // 4:3 * mask_height // 4, mask_width // 2] = 255
    image_path = images_dir / name
    mask_path = masks_dir / name
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


class _TrackingTransform:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
        self.called = True
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask.copy()),
        }


class FivesTests(unittest.TestCase):
    def test_centered_coordinates_are_exactly_four(self) -> None:
        self.assertEqual(
            centered_fives_coordinates(2048, 2048, 512),
            [(512, 512), (1024, 512), (512, 1024), (1024, 1024)],
        )
        with self.assertRaisesRegex(ValueError, "centered 2x2"):
            centered_fives_coordinates(900, 2048, 512)

    def test_discovery_and_records_are_complete_fixed_center_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path, mask_path = _write_pair(root)
            pairs = discover_fives_pairs(
                [".png"], images_dir=root / "Original", masks_dir=root / "Ground truth"
            )
            records = build_fives_patch_records(pairs, patch_size=8)

        self.assertEqual(pairs, [(image_path, mask_path)])
        self.assertEqual(len(records), 4)
        self.assertEqual([(record.x, record.y) for record in records], [(8, 8), (16, 8), (8, 16), (16, 16)])
        self.assertTrue(all(record.scale == 1.0 for record in records))
        self.assertTrue(all(record.scale_label == "fives_center" for record in records))
        self.assertTrue(all(record.source_id == "FIVES/sample.png" for record in records))

    def test_discovery_rejects_incomplete_pairs_and_dimension_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path, mask_path = _write_pair(root)
            mask_path.unlink()
            with self.assertRaisesRegex(ValueError, "missing masks"):
                discover_fives_pairs(
                    [".png"], images_dir=root / "Original", masks_dir=root / "Ground truth"
                )
            _write_pair(root, mask_size=(16, 32))
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                build_fives_patch_records([(image_path, root / "Ground truth" / "sample.png")], 8)

    def test_dataset_applies_transform_and_maps_multiclass_vessels_to_loci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair = _write_pair(root)
            records = build_fives_patch_records([pair], patch_size=8)
            tracking_transform = _TrackingTransform()
            binary_dataset = FivesPatchDataset(records, 127, tracking_transform)
            binary_sample = binary_dataset[0]
            multiclass_dataset = FivesPatchDataset([records[1]], 127, segmentation_mode="multiclass")
            multiclass_sample = multiclass_dataset[0]

        self.assertTrue(tracking_transform.called)
        self.assertEqual(tuple(binary_sample["mask"].shape), (1, 8, 8))
        self.assertEqual(tuple(multiclass_sample["mask"].shape), (8, 8))
        self.assertEqual(multiclass_sample["mask"].dtype, torch.long)
        self.assertEqual(set(multiclass_sample["mask"].unique().tolist()), {0, 1})

    def test_switch_defaults_off_and_training_composition_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("project:\n  name: test\n", encoding="utf-8")
            config = load_config(config_path)
        self.assertFalse(config["data"]["use_fives"])
        self.assertEqual(load_fives_training_records(config), [])

        fungal = TensorDataset(torch.zeros(2, 1))
        fives = TensorDataset(torch.zeros(4, 1))
        self.assertIs(combine_training_datasets(fungal, None), fungal)
        self.assertEqual(len(combine_training_datasets(fungal, fives)), 6)

    def test_all_tracked_configs_expose_disabled_switch(self) -> None:
        for path in (
            "config.yaml",
            "config_segformer_mit_b3.yaml",
            "multiclass-config.yaml",
            "multiclass-segformer-config.yaml",
            "multiclass-segformer-mit-b3-geometry-config.yaml",
        ):
            with self.subTest(path=path):
                self.assertFalse(load_config(path)["data"]["use_fives"])

    def test_visualization_writes_one_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair = _write_pair(root)
            records = build_fives_patch_records([pair], patch_size=8)
            output_path = root / "visualization.png"
            result = create_fives_patch_visualization(*pair, records, output_path)
            self.assertEqual(result, output_path)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
