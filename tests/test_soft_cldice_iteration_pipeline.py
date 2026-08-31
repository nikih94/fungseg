from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.add_soft_cldice_iteration_margin import add_iteration_margin
from src.build_soft_cldice_iteration_map import build_iteration_rows
from src.data.soft_cldice_iterations import map_training_iterations_to_sources
from src.losses.combined import SoftCLDiceLoss
from src.metrics.segmentation import soft_cldice_scores_from_probabilities
from src.patching import OriginalImageRecord
from src.utils.io import save_csv


class SoftCLDiceIterationPipelineTests(unittest.TestCase):
    def test_build_then_margin_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask_dir = root / "masks"
            mask_dir.mkdir()
            thick = np.zeros((11, 13), dtype=np.uint8)
            thick[2:9, 3:10] = 255
            Image.fromarray(thick).save(mask_dir / "thick.png")
            Image.fromarray(np.zeros((5, 7), dtype=np.uint8)).save(
                mask_dir / "empty.png"
            )

            required_rows = build_iteration_rows(mask_dir)
            by_name = {row["mask_filename"]: row for row in required_rows}
            self.assertEqual(by_name["thick.png"]["required_iterations"], 3)
            self.assertEqual(by_name["empty.png"]["required_iterations"], 0)

            required_csv = root / "required.csv"
            save_csv(required_csv, required_rows)
            adjusted = add_iteration_margin(
                required_csv,
                margin_iterations=7,
                round_up_to=10,
                minimum_iterations=10,
                maximum_iterations=12,
            )
            adjusted_by_name = {row["mask_filename"]: row for row in adjusted}
            self.assertEqual(adjusted_by_name["thick.png"]["training_iterations"], 10)
            self.assertEqual(adjusted_by_name["empty.png"]["training_iterations"], 10)
            self.assertEqual(adjusted_by_name["thick.png"]["round_up_to"], 10)

    def test_margin_rounds_up_to_iteration_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "required.csv"
            save_csv(
                csv_path,
                [
                    {
                        "mask_filename": "already-aligned.png",
                        "mask_stem": "already-aligned",
                        "required_iterations": 20,
                    },
                    {
                        "mask_filename": "rounded.png",
                        "mask_stem": "rounded",
                        "required_iterations": 21,
                    },
                ],
            )

            adjusted = add_iteration_margin(
                csv_path,
                margin_iterations=10,
                round_up_to=10,
            )

            self.assertEqual(
                [row["training_iterations"] for row in adjusted],
                [30, 40],
            )

    def test_rounding_bucket_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "required.csv"
            save_csv(
                csv_path,
                [{"required_iterations": 3}],
            )
            with self.assertRaisesRegex(ValueError, "round_up_to"):
                add_iteration_margin(
                    csv_path,
                    margin_iterations=10,
                    round_up_to=0,
                )

    def test_adjusted_csv_maps_mask_names_to_image_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iterations.csv"
            rows = [
                {
                    "mask_filename": "sample-mask.png",
                    "mask_stem": "sample-mask",
                    "training_iterations": 17,
                }
            ]
            save_csv(csv_path, rows)
            record = OriginalImageRecord(
                source_id="sample-image.jpg",
                image_path=Path("sample-image.jpg"),
                mask_path=Path("sample-mask.png"),
                width=20,
                height=10,
            )
            self.assertEqual(
                map_training_iterations_to_sources(csv_path, [record]),
                {"sample-image.jpg": 17},
            )

    def test_variable_scores_and_gradients_match_individual_calls(self) -> None:
        torch.manual_seed(9)
        predictions = torch.rand(3, 1, 17, 19)
        targets = (torch.rand(3, 1, 17, 19) > 0.6).float()
        iterations = torch.tensor([0, 2, 4])

        grouped = soft_cldice_scores_from_probabilities(
            predictions, targets, iterations=iterations
        )
        individual = torch.stack(
            [
                soft_cldice_scores_from_probabilities(
                    predictions[index : index + 1],
                    targets[index : index + 1],
                    iterations=int(iterations[index]),
                )[0]
                for index in range(3)
            ]
        )
        torch.testing.assert_close(grouped, individual)

        logits = torch.randn(3, 1, 17, 19, requires_grad=True)
        loss = SoftCLDiceLoss(iterations=99)(
            logits, targets, soft_cldice_iterations=iterations
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mapping_rejects_missing_mask_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iterations.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "mask_filename",
                        "mask_stem",
                        "training_iterations",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mask_filename": "other.png",
                        "mask_stem": "other",
                        "training_iterations": 4,
                    }
                )
            record = OriginalImageRecord(
                source_id="missing.jpg",
                image_path=Path("missing.jpg"),
                mask_path=Path("missing.png"),
                width=1,
                height=1,
            )
            with self.assertRaisesRegex(ValueError, "missing.jpg"):
                map_training_iterations_to_sources(csv_path, [record])


if __name__ == "__main__":
    unittest.main()
