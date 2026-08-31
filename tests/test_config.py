from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.train import segmentation_summary_metadata
from src.utils.config import config_for_persistence, load_config, resolve_mask_dir


class ConfigTests(unittest.TestCase):
    def test_evaluation_only_join_masks_are_retained_for_test_evaluation(self) -> None:
        config = load_config("multiclass-config.yaml")

        persisted = config_for_persistence(config)

        self.assertEqual(
            persisted["join_masks"],
            {
                "enabled": False,
                "masks_dir": "data/join_masks",
                "merge_with_loci": False,
                "evaluation_enabled": True,
            },
        )
        self.assertEqual(config["split"]["mode"], "csv_kfold")
        self.assertEqual(persisted["split"]["mode"], "csv_kfold")
        self.assertEqual(persisted["split"]["csv_path"], "data/image_splits.csv")
        self.assertEqual(persisted["cv"]["n_splits"], 5)
        self.assertEqual(persisted["cv"]["random_state"], 42)
        self.assertGreaterEqual(persisted["validation"]["start_epoch"], 1)
        self.assertLessEqual(
            persisted["validation"]["start_epoch"],
            persisted["train"]["epochs"],
        )
        self.assertGreater(
            persisted["validation"]["full_image"]["batch_size"], 0
        )
        self.assertEqual(config["train"]["monitor"], "val_dice_cldice_per_image")
        self.assertEqual(
            config["scheduler"]["monitor"],
            "val_dice_cldice_per_image",
        )
        self.assertEqual(
            config["validation"]["full_image"]["monitor"],
            {"dice_weight": 0.6, "cldice_weight": 0.4},
        )
        self.assertFalse(config["train"]["best_interval_checkpoint"]["enabled"])
        self.assertFalse(config["train"]["save_last_checkpoint"])
        self.assertFalse(config["qualitative_evaluation"]["enabled"])


    def test_b2_refinement_config_persists_only_its_model_options(self) -> None:
        config = load_config("multiclass-segformer-mit-b2-refinement-config.yaml")

        persisted = config_for_persistence(config)

        self.assertEqual(
            persisted["join_masks"],
            {
                "enabled": True,
                "masks_dir": "data/join_masks",
                "merge_with_loci": True,
            },
        )
        self.assertEqual(
            persisted["model"],
            {
                "name": "segformer_mit_b2_refinement",
                "in_channels": 3,
                "num_classes": 3,
                "encoder_name": "mit_b2",
                "encoder_weights": "imagenet",
                "encoder_depth": 5,
                "decoder_segmentation_channels": 256,
                "shallow_channels": [16, 32],
                "refine_half_channels": [128, 64],
                "refine_full_channels": [32, 32],
            },
        )

    def test_refinement_config_persists_only_its_model_options(self) -> None:
        config = load_config("multiclass-segformer-mit-b1-refinement-config.yaml")

        persisted = config_for_persistence(config)

        self.assertEqual(
            persisted["model"],
            {
                "name": "segformer_mit_b1_refinement",
                "in_channels": 3,
                "num_classes": 3,
                "encoder_name": "mit_b1",
                "encoder_weights": "imagenet",
                "encoder_depth": 5,
                "decoder_segmentation_channels": 256,
                "shallow_channels": [16, 32],
                "refine_half_channels": [128, 64],
                "refine_full_channels": [32, 32],
            },
        )
        self.assertNotIn("decoder_normalization", persisted["model"])
        self.assertNotIn("decoder_channels", persisted["model"])
        self.assertNotIn("decoder_attention_type", persisted["model"])
        self.assertNotIn("upsampling", persisted["model"])

    def test_resolves_configured_segmentation_target_mask_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "segmentation:",
                        "  target: inoculum",
                        "paths:",
                        "  images_dir: data/images",
                        "  mask_dirs:",
                        "    loci: data/loci_masks",
                        "    inoculum: data/inoculum_masks",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(resolve_mask_dir(config), Path("data/inoculum_masks"))

    def test_legacy_masks_dir_is_preserved_for_old_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        "  images_dir: data/images-small",
                        "  masks_dir: data/masks-small",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config["segmentation"]["target"], "legacy")
        self.assertEqual(resolve_mask_dir(config), Path("data/masks-small"))

    def test_persisted_multiclass_segformer_config_contains_only_relevant_options(self) -> None:
        config = load_config("multiclass-segformer-config.yaml")

        persisted = config_for_persistence(config, training_date="2026-07-29")

        self.assertEqual(persisted["training_date"], "2026-07-29")
        self.assertNotIn("target", persisted["segmentation"])
        self.assertNotIn("masks_dir", persisted["paths"])
        self.assertNotIn("decoder_normalization", persisted["model"])
        self.assertNotIn("decoder_channels", persisted["model"])
        self.assertNotIn("decoder_attention_type", persisted["model"])
        self.assertEqual(
            set(persisted["loss"]),
            {
                "name",
                "cross_entropy_weight",
                "dice_weight",
                "loci_cldice_weight",
                "iterations",
                "iterations_csv",
                "smooth",
                "cldice_smooth",
            },
        )
        self.assertNotIn("threshold", persisted["train"])
        self.assertNotIn("threshold", persisted["inference"])
        self.assertNotIn("threshold_start", persisted["test_evaluation"])
        self.assertNotIn("threshold_stop", persisted["test_evaluation"])
        self.assertNotIn("threshold_step", persisted["test_evaluation"])
        self.assertNotIn("threshold_sweep", persisted["test_evaluation"])
        self.assertNotIn("cv", persisted)
        self.assertEqual(persisted["validation"], config["validation"])

        self.assertIn("bce_weight", config["loss"])
        self.assertIn("decoder_channels", config["model"])

    def test_geometry_config_persists_geometry_loss_and_validation_subset(self) -> None:
        config = load_config("multiclass-segformer-mit-b3-geometry-config.yaml")

        persisted = config_for_persistence(config)

        self.assertEqual(config["model"]["encoder_name"], "mit_b3")
        self.assertEqual(
            persisted["loss"]["name"],
            "multiclass_geometry_ce_dice_loci_cldice",
        )
        self.assertEqual(
            persisted["loss"]["geometry_aware_ce"]["separator_radius_multipliers"],
            [0.5, 1.0, 1.5],
        )
        self.assertEqual(
            persisted["validation"]["full_image"],
            {
                "enabled": False,
                "batch_size": 1,
                "interval_epochs": 1,
                "selection": "smallest_area",
                "max_images": 3,
                "monitor": {"dice_weight": 0.5, "cldice_weight": 0.5},
            },
        )

    def test_smallest_area_validation_requires_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "train:\n  full_image_validation:\n    selection: smallest_area\n    max_images: 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "max_images must be positive"):
                load_config(config_path)

    def test_full_image_validation_batch_size_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "validation:\n  full_image:\n    batch_size: 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
                load_config(config_path)

    def test_validation_start_epoch_must_not_exceed_training_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "validation:\n  start_epoch: 3\ntrain:\n  epochs: 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not exceed train.epochs"):
                load_config(config_path)

    def test_checked_in_configs_enable_background_patches(self) -> None:
        config_paths = (
            "config.yaml",
            "config_segformer_mit_b3.yaml",
            "multiclass-config.yaml",
            "multiclass-segformer-config.yaml",
            "multiclass-segformer-mit-b1-refinement-config.yaml",
            "multiclass-segformer-mit-b2-refinement-config.yaml",
            "multiclass-segformer-mit-b3-geometry-config.yaml",
        )
        for config_path in config_paths:
            with self.subTest(config=config_path):
                background_config = load_config(config_path)["patching"]["train"][
                    "background_only"
                ]
                self.assertTrue(background_config["enabled"])
                percentage = background_config["percentage_of_foreground"]
                self.assertGreaterEqual(percentage, 0.0)
                self.assertLessEqual(percentage, 100.0)

    def test_background_patch_percentage_must_be_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "patching:\n"
                "  train:\n"
                "    background_only:\n"
                "      enabled: true\n"
                "      percentage_of_foreground: 101\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                load_config(config_path)

    def test_validation_section_is_persisted_with_patch_monitor(self) -> None:
        config = load_config("config.yaml")
        config["train"]["monitor"] = "val_dice_per_patch"
        config["scheduler"] = {"name": "none"}

        persisted = config_for_persistence(config)

        self.assertNotIn("full_image_monitor", persisted["train"])
        self.assertIn("full_image", persisted["validation"])

    def test_persisted_binary_config_removes_multiclass_defaults(self) -> None:
        config = load_config("config.yaml")

        persisted = config_for_persistence(config)

        self.assertNotIn("classes", persisted["segmentation"])
        self.assertNotIn("overlap_precedence", persisted["segmentation"])
        self.assertNotIn("decision", persisted["inference"])
        self.assertEqual(
            set(persisted["paths"]["mask_dirs"]),
            {persisted["segmentation"]["target"]},
        )

    def test_multiclass_summary_metadata_reports_mode_and_both_mask_dirs(self) -> None:
        config = load_config("multiclass-segformer-config.yaml")

        metadata = segmentation_summary_metadata(config, Path("data/loci_masks"))

        self.assertEqual(metadata["segmentation_mode"], "multiclass")
        self.assertIsNone(metadata["segmentation_target"])
        self.assertIsNone(metadata["mask_dir"])
        self.assertEqual(
            metadata["mask_dirs"],
            {
                "loci": "data/loci_masks",
                "inoculum": "data/inoculum_masks",
            },
        )


if __name__ == "__main__":
    unittest.main()
