from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inference.in_folder import list_recursive_input_images, mask_output_path


class InFolderInferenceTests(unittest.TestCase):
    def test_list_recursive_input_images_finds_supported_nested_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()

            first_image = root / "first.tif"
            second_image = nested / "second.PNG"
            ignored_text = nested / "notes.txt"
            generated_mask = nested / "second_mask.png"

            first_image.touch()
            second_image.touch()
            ignored_text.touch()
            generated_mask.touch()

            images = list_recursive_input_images(root, [".tif", ".png"])

        self.assertEqual(images, [first_image, second_image])

    def test_mask_output_path_preserves_location_and_appends_mask_suffix(self) -> None:
        image_path = Path("folder/sub/sample.tif")

        self.assertEqual(mask_output_path(image_path), Path("folder/sub/sample_mask.png"))


if __name__ == "__main__":
    unittest.main()
