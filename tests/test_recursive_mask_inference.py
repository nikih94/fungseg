from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.inference.recursive_masks import (
    default_output_dir,
    list_recursive_input_images,
    masks_from_probabilities,
    output_path,
    probability_maps_from_probabilities,
    run_recursive_mask_inference,
    validate_no_output_collisions,
)


class RecursiveMaskInferenceTests(unittest.TestCase):
    def test_lists_nested_images_and_skips_old_masks_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            source = nested / "sample.tif"
            old_mask = nested / "sample_MASK.PNG"
            source.touch()
            old_mask.touch()
            (nested / "notes.txt").touch()

            images = list_recursive_input_images(root, [".tif", ".png"])

        self.assertEqual(images, [source])

    def test_derives_sibling_output_and_preserves_relative_path(self) -> None:
        input_dir = Path("/data/fung-all-images")
        output_dir = default_output_dir(input_dir)
        image_path = input_dir / "group/nested/sample.tif"

        self.assertEqual(output_dir, Path("/data/fung-all-images_masks"))
        self.assertEqual(
            output_path(image_path, input_dir, output_dir, "inoculum"),
            Path("/data/fung-all-images_masks/group/nested/sample_inoculum.png"),
        )

    def test_rejects_sources_that_would_overwrite_the_same_masks(self) -> None:
        input_dir = Path("/data/images")
        images = [input_dir / "sample.tif", input_dir / "sample.png"]

        with self.assertRaisesRegex(ValueError, "overwrite the same output masks"):
            validate_no_output_collisions(images, input_dir)

    def test_multiclass_probabilities_create_combined_and_binary_class_masks(self) -> None:
        probabilities = np.array(
            [
                [[0.9, 0.1], [0.1, 0.1]],
                [[0.05, 0.8], [0.2, 0.1]],
                [[0.05, 0.1], [0.7, 0.8]],
            ],
            dtype=np.float32,
        )
        config = {
            "segmentation": {
                "mode": "multiclass",
                "classes": {"background": 0, "loci": 1, "inoculum": 2},
            }
        }

        masks = masks_from_probabilities(probabilities, config)

        np.testing.assert_array_equal(masks["mask"], np.array([[0, 1], [2, 2]], dtype=np.uint8))
        np.testing.assert_array_equal(
            masks["loci"], np.array([[0, 255], [0, 0]], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            masks["inoculum"], np.array([[0, 0], [255, 255]], dtype=np.uint8)
        )

    def test_binary_probabilities_create_combined_and_target_mask(self) -> None:
        config = {
            "segmentation": {"mode": "binary", "target": "loci"},
            "inference": {"threshold": 0.5},
        }

        masks = masks_from_probabilities(
            np.array([[0.4, 0.5]], dtype=np.float32),
            config,
        )

        self.assertEqual(set(masks), {"mask", "loci"})
        np.testing.assert_array_equal(masks["mask"], np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(masks["loci"], masks["mask"])

    def test_multiclass_probability_maps_are_scaled_to_grayscale(self) -> None:
        probabilities = np.array(
            [
                [[0.9, 0.1]],
                [[0.05, 0.8]],
                [[0.05, 0.1]],
            ],
            dtype=np.float32,
        )
        config = {
            "segmentation": {
                "mode": "multiclass",
                "classes": {"background": 0, "loci": 1, "inoculum": 2},
            },
            "inference": {"save_probabilities": True},
        }

        probability_maps = probability_maps_from_probabilities(probabilities, config)

        self.assertEqual(set(probability_maps), {"prob_loci", "prob_inoculum"})
        np.testing.assert_allclose(
            probability_maps["prob_loci"],
            np.array([[12.75, 204.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            probability_maps["prob_inoculum"],
            np.array([[12.75, 25.5]], dtype=np.float32),
        )

    def test_probability_maps_are_disabled_by_config(self) -> None:
        config = {
            "segmentation": {"mode": "multiclass"},
            "inference": {"save_probabilities": False},
        }

        probability_maps = probability_maps_from_probabilities(
            np.zeros((3, 1, 1), dtype=np.float32),
            config,
        )

        self.assertEqual(probability_maps, {})

    def test_run_writes_multiclass_masks_and_probability_maps_in_mirrored_tree(self) -> None:
        probabilities = np.array(
            [
                [[0.9, 0.1]],
                [[0.05, 0.8]],
                [[0.05, 0.1]],
            ],
            dtype=np.float32,
        )
        config = {
            "segmentation": {
                "mode": "multiclass",
                "classes": {"background": 0, "loci": 1, "inoculum": 2},
            },
            "data": {"image_extensions": [".tif", ".png"]},
            "train": {"device": "cpu"},
            "model": {},
            "training_date": "2026-07-28",
            "inference": {"threshold": 0.5, "save_probabilities": True},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_dir = temp_root / "images"
            nested = input_dir / "group"
            nested.mkdir(parents=True)
            source = nested / "sample.tif"
            old_mask = nested / "sample_mask.png"
            source.touch()
            old_mask.touch()

            with (
                patch("src.inference.recursive_masks.load_config", return_value=config),
                patch("src.inference.recursive_masks.build_model", return_value=torch.nn.Identity()),
                patch("src.inference.recursive_masks.load_checkpoint"),
                patch(
                    "src.inference.recursive_masks.predict_probabilities_on_image",
                    return_value=probabilities,
                ),
            ):
                count, output_dir = run_recursive_mask_inference(
                    "config.yaml",
                    "best.pt",
                    input_dir,
                )

            files = sorted(path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file())
            saved_config_text = (output_dir / "config.yaml").read_text()
            combined = np.array(Image.open(output_dir / "group/sample_mask.png"))
            loci = np.array(Image.open(output_dir / "group/sample_loci.png"))
            inoculum = np.array(Image.open(output_dir / "group/sample_inoculum.png"))
            prob_loci = np.array(Image.open(output_dir / "group/sample_prob_loci.png"))
            prob_inoculum = np.array(Image.open(output_dir / "group/sample_prob_inoculum.png"))

        self.assertEqual(count, 1)
        self.assertEqual(
            files,
            [
                Path("config.yaml"),
                Path("group/sample_inoculum.png"),
                Path("group/sample_loci.png"),
                Path("group/sample_mask.png"),
                Path("group/sample_prob_inoculum.png"),
                Path("group/sample_prob_loci.png"),
            ],
        )
        self.assertIn("mode: multiclass", saved_config_text)
        self.assertIn("training_date: '2026-07-28'", saved_config_text)
        np.testing.assert_array_equal(combined, np.array([[0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(loci, np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(inoculum, np.array([[0, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(prob_loci, np.array([[12, 204]], dtype=np.uint8))
        np.testing.assert_array_equal(prob_inoculum, np.array([[12, 25]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
