from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inference.other_test_data_evaluation import list_input_images, result_path


class OtherTestDataEvaluationTests(unittest.TestCase):
    def test_lists_nested_images_and_skips_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_dir = root / "paper_a"
            results_dir = root / "results"
            paper_dir.mkdir()
            results_dir.mkdir()
            (paper_dir / "image.tif").touch()
            (paper_dir / "notes.txt").touch()
            (results_dir / "paper_a").mkdir()
            (results_dir / "paper_a" / "image_overlay.png").touch()

            images = list_input_images(root, results_dir, [".tif", ".png"])

        self.assertEqual(images, [paper_dir / "image.tif"])

    def test_result_path_preserves_input_structure(self) -> None:
        input_dir = Path("data/other-test-data")
        results_dir = Path("data/other-test-data/results")
        image_path = input_dir / "paper_a/nested/sample.tif"

        self.assertEqual(
            result_path(image_path, input_dir, results_dir, "mask"),
            results_dir / "paper_a/nested/sample_mask.png",
        )


if __name__ == "__main__":
    unittest.main()
