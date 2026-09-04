from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data.patch_cache import (
    CachedSegmentationPatchDataset,
    build_epoch_training_crop_records,
    build_static_patch_cache,
    build_static_validation_patch_cache,
)
from src.patching import OriginalImageRecord


def patching_config(
    *,
    filter_empty: bool = False,
    background_percentage: float = 10.0,
) -> dict:
    return {
        "patch_size": 4,
        "overlap": 2,
        "stride": 4,
        "filter_empty_patches": filter_empty,
        "mask_threshold": 127,
        "min_foreground_pixels": 1,
        "train": {
            "random_offset": {
                "enabled": True,
                "max_fraction_of_patch": 0.5,
            },
            "background_only": {
                "enabled": True,
                "percentage_of_foreground": background_percentage,
            },
            "scaled_context": {"enabled": False},
        },
    }


def build_binary_cache(
    root: Path,
    image: np.ndarray,
    mask: np.ndarray,
    config: dict,
    *,
    iterations: bool = True,
):
    image_path, mask_path = root / "image.png", root / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    original = OriginalImageRecord(
        "image.png",
        image_path,
        mask_path,
        width=image.shape[1],
        height=image.shape[0],
    )
    return build_static_patch_cache(
        [original],
        root,
        config,
        segmentation_mode="binary",
        merge_join_masks=False,
        compute_soft_cldice_iterations=iterations,
        iteration_margin=10,
        iteration_round_up_to=10,
    )


class StaticPatchCacheTests(unittest.TestCase):
    def test_cache_geometry_epoch_randomization_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yy, xx = np.indices((12, 12))
            image = np.stack(
                [
                    xx.astype(np.uint8),
                    yy.astype(np.uint8),
                    np.full((12, 12), 7, dtype=np.uint8),
                ],
                axis=-1,
            )
            mask = ((xx + yy) % 3 == 0).astype(np.uint8) * 255
            config = patching_config()
            cache = build_binary_cache(root, image, mask, config)
            first = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=1,
                base_seed=9,
                fold_index=0,
                segmentation_mode="binary",
                merge_join_masks=False,
            )
            repeated = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=1,
                base_seed=9,
                fold_index=0,
                segmentation_mode="binary",
                merge_join_masks=False,
            )
            second = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=2,
                base_seed=9,
                fold_index=0,
                segmentation_mode="binary",
                merge_join_masks=False,
            )

            cached_images = np.load(
                cache.cache_dir / "images.npy", mmap_mode="r"
            )
            self.assertEqual(cached_images.shape[1:3], (6, 6))
            self.assertEqual(
                [(item.cache_index, item.x, item.y) for item in first],
                [(item.cache_index, item.x, item.y) for item in repeated],
            )
            self.assertNotEqual(
                [(item.x, item.y) for item in first],
                [(item.x, item.y) for item in second],
            )
            static_by_index = {
                record.cache_index: record for record in cache.records
            }
            for crop in first:
                static = static_by_index[crop.cache_index]
                final_x = max(0, static.source_width - crop.patch_size)
                final_y = max(0, static.source_height - crop.patch_size)
                if static.anchor_x in {0, final_x}:
                    self.assertEqual(crop.x, static.anchor_x)
                if static.anchor_y in {0, final_y}:
                    self.assertEqual(crop.y, static.anchor_y)
                self.assertGreaterEqual(crop.soft_cldice_iterations, 10)
                self.assertEqual(crop.soft_cldice_iterations % 10, 0)

            dataset = CachedSegmentationPatchDataset(
                first,
                cache,
                None,
                "binary",
                127,
                False,
                None,
                None,
                3,
            )
            with patch("src.data.patch_cache.Image.open") as open_mock:
                for index, crop in enumerate(first):
                    sample = dataset[index]
                    expected_image = torch.from_numpy(
                        image[
                            crop.y : crop.y + crop.patch_size,
                            crop.x : crop.x + crop.patch_size,
                        ].copy()
                    ).permute(2, 0, 1).float() / 255.0
                    expected_mask = torch.from_numpy(
                        (
                            mask[
                                crop.y : crop.y + crop.patch_size,
                                crop.x : crop.x + crop.patch_size,
                            ]
                            > 127
                        ).astype(np.float32)
                    ).unsqueeze(0)
                    self.assertTrue(
                        torch.equal(sample["image"], expected_image)
                    )
                    self.assertTrue(
                        torch.equal(sample["mask"], expected_mask)
                    )
                open_mock.assert_not_called()
            cache.cleanup()
            self.assertFalse((root / ".train_patch_cache").exists())

    def test_final_crop_controls_filter_and_background_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.zeros((20, 20, 3), dtype=np.uint8)
            mask = np.ones((20, 20), dtype=np.uint8) * 255
            mask[:4] = 0
            config = patching_config(filter_empty=True)
            config["train"]["random_offset"]["enabled"] = False
            cache = build_binary_cache(
                root, image, mask, config, iterations=False
            )
            records = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=1,
                base_seed=4,
                fold_index=0,
                segmentation_mode="binary",
                merge_join_masks=False,
            )
            self.assertEqual(
                sum(not item.is_background_only for item in records), 20
            )
            self.assertEqual(
                sum(item.is_background_only for item in records), 2
            )

            cache.cleanup()
            mask[:] = 0
            mask[4:6] = 255
            cache = build_binary_cache(root, image, mask, config)
            config["train"]["background_only"]["enabled"] = False
            records = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=1,
                base_seed=4,
                fold_index=0,
                segmentation_mode="binary",
                merge_join_masks=False,
            )
            self.assertGreater(cache.records[0].soft_cldice_iterations, 0)
            self.assertFalse(
                any(item.cache_index == 0 for item in records)
            )
            cache.cleanup()

    def test_validation_cache_uses_exact_grid_and_adaptive_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yy, xx = np.indices((6, 6))
            image = np.stack(
                [
                    xx.astype(np.uint8),
                    yy.astype(np.uint8),
                    np.full((6, 6), 9, dtype=np.uint8),
                ],
                axis=-1,
            )
            mask = np.zeros((6, 6), dtype=np.uint8)
            mask[2:4, 1:5] = 255
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            original = OriginalImageRecord(
                "image.png", image_path, mask_path, 6, 6
            )
            config = patching_config()
            config["stride"] = 2

            cache = build_static_validation_patch_cache(
                [original],
                root,
                config,
                segmentation_mode="binary",
                merge_join_masks=False,
                compute_soft_cldice_iterations=True,
                iteration_margin=4,
                iteration_round_up_to=2,
            )

            self.assertEqual(
                [(item.x, item.y) for item in cache.records],
                [(0, 0), (2, 0), (0, 2), (2, 2)],
            )
            images, targets = cache.arrays()
            self.assertEqual(targets.shape, (1, 4, 4))
            for record in cache.records:
                np.testing.assert_array_equal(
                    images[record.cache_index],
                    image[
                        record.y : record.y + 4,
                        record.x : record.x + 4,
                    ],
                )
                if record.loss_target_index is None:
                    self.assertIsNone(record.soft_cldice_iterations)
                    continue
                np.testing.assert_array_equal(
                    targets[record.loss_target_index],
                    (
                        mask[
                            record.y : record.y + 4,
                            record.x : record.x + 4,
                        ]
                        > 127
                    ).astype(np.uint8),
                )
                self.assertGreaterEqual(record.soft_cldice_iterations, 4)
                self.assertEqual(record.soft_cldice_iterations % 2, 0)
            np.testing.assert_array_equal(
                cache.full_target("image.png"), (mask > 127).astype(np.uint8)
            )
            cache.cleanup()
            self.assertFalse((root / ".validation_patch_cache").exists())

    def test_multiclass_masks_share_crop_and_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.full((8, 8, 3), 64, dtype=np.uint8)
            loci = np.zeros((8, 8), dtype=np.uint8)
            inoculum = np.zeros((8, 8), dtype=np.uint8)
            join = np.zeros((8, 8), dtype=np.uint8)
            loci[1:4, 1:4] = 255
            inoculum[2:4, 2:4] = 255
            join[0, 3] = 255
            paths = {}
            for name, array in (
                ("image", image),
                ("loci", loci),
                ("inoculum", inoculum),
                ("join", join),
            ):
                paths[name] = root / f"{name}.png"
                Image.fromarray(array).save(paths[name])
            original = OriginalImageRecord(
                "image.png",
                paths["image"],
                paths["loci"],
                8,
                8,
                mask_paths={
                    name: paths[name]
                    for name in ("loci", "inoculum", "join")
                },
            )
            config = patching_config()
            config["train"]["random_offset"]["enabled"] = False
            cache = build_static_patch_cache(
                [original],
                root,
                config,
                segmentation_mode="multiclass",
                merge_join_masks=True,
                compute_soft_cldice_iterations=True,
                iteration_margin=10,
                iteration_round_up_to=10,
            )
            records = build_epoch_training_crop_records(
                cache,
                ["image.png"],
                config,
                epoch=1,
                base_seed=1,
                fold_index=0,
                segmentation_mode="multiclass",
                merge_join_masks=True,
            )
            dataset = CachedSegmentationPatchDataset(
                records,
                cache,
                None,
                "multiclass",
                127,
                True,
                None,
                None,
                3,
            )
            first = dataset[0]
            self.assertEqual(first["mask"].shape, (4, 4))
            self.assertEqual(int(first["mask"][0, 3]), 1)
            self.assertEqual(int(first["mask"][2, 2]), 2)
            self.assertEqual(first["overlap_pixels"], 4)
            cache.cleanup()

            validation_cache = build_static_validation_patch_cache(
                [original],
                root,
                config,
                segmentation_mode="multiclass",
                merge_join_masks=True,
                compute_soft_cldice_iterations=True,
                iteration_margin=4,
                iteration_round_up_to=2,
            )
            full_target = validation_cache.full_target("image.png")
            self.assertEqual(int(full_target[0, 3]), 1)
            self.assertEqual(int(full_target[2, 2]), 2)
            np.testing.assert_array_equal(
                validation_cache.full_join_mask("image.png"), join
            )
            validation_cache.cleanup()


if __name__ == "__main__":
    unittest.main()
