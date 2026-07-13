from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference import load_rgb_image


class InferenceImageLoadingTests(unittest.TestCase):
    def test_loads_uncompressed_16bit_tiff_as_scaled_rgb(self) -> None:
        source = np.array([[0, 256], [32768, 65535]], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "raw_16bit.tif"
            Image.fromarray(source).save(image_path, compression="raw")
            loaded = load_rgb_image(image_path)

        expected_channel = (source >> 8).astype(np.uint8)
        expected = np.repeat(expected_channel[..., np.newaxis], 3, axis=2)
        np.testing.assert_array_equal(loaded, expected)


if __name__ == "__main__":
    unittest.main()
