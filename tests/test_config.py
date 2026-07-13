from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.utils.config import load_config, resolve_mask_dir


class ConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
