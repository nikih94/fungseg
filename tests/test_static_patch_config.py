from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.config import config_for_persistence, load_config


class StaticPatchConfigTests(unittest.TestCase):
    def test_active_config_enables_static_iterations(self) -> None:
        config = load_config("multiclass-config.yaml")
        self.assertTrue(config["data"]["train_patch_cache"]["enabled"])
        self.assertIsNone(config["loss"]["iterations_csv"])
        self.assertEqual(
            config["loss"]["static_patch_iterations"],
            {
                "enabled": True,
                "margin_iterations": 10,
                "round_up_to": 10,
            },
        )
        self.assertIn(
            "static_patch_iterations", config_for_persistence(config)["loss"]
        )

    def test_scaled_context_and_static_cache_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "data:\n"
                "  train_patch_cache:\n"
                "    enabled: true\n"
                "patching:\n"
                "  overlap: 128\n"
                "  train:\n"
                "    scaled_context:\n"
                "      enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_config(config_path)

    def test_checked_in_scaled_context_configs_disable_static_cache(self) -> None:
        paths = (
            "config.yaml",
            "config_segformer_mit_b3.yaml",
            "multiclass-segformer-config.yaml",
            "multiclass-segformer-mit-b1-refinement-config.yaml",
            "multiclass-segformer-mit-b2-refinement-config.yaml",
            "multiclass-segformer-mit-b3-geometry-config.yaml",
        )
        for path in paths:
            with self.subTest(path=path):
                config = load_config(path)
                self.assertTrue(
                    config["patching"]["train"]["scaled_context"]["enabled"]
                )
                self.assertFalse(
                    config["data"]["train_patch_cache"]["enabled"]
                )

    def test_iteration_csv_overrides_persisted_static_settings(self) -> None:
        config = load_config("multiclass-config.yaml")
        config["loss"]["iterations_csv"] = "iterations.csv"
        persisted = config_for_persistence(config)
        self.assertNotIn("static_patch_iterations", persisted["loss"])


if __name__ == "__main__":
    unittest.main()
