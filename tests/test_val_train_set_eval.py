from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.val_train_set_eval import run_val_train_set_evaluation


class ValTrainSetEvaluationTests(unittest.TestCase):
    def test_writes_split_means_combined_mean_and_only_split_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            for name, foreground in (("train", True), ("val", False), ("test", False)):
                image = np.full((4, 5, 3), 120, dtype=np.uint8)
                mask = np.zeros((4, 5), dtype=np.uint8)
                if foreground:
                    mask[:2, :2] = 255
                Image.fromarray(image).save(images / f"{name}.png")
                Image.fromarray(mask).save(masks / f"{name}.png")
            split_path = root / "splits.csv"
            split_path.write_text(
                "filename,split\ntrain.png,train\nval.png,validation\ntest.png,test\n",
                encoding="utf-8",
            )
            config = {
                "paths": {"images_dir": str(images), "mask_dirs": {"loci": str(masks)}},
                "segmentation": {"target": "loci"},
                "data": {"image_extensions": [".png"]},
                "split": {"mode": "csv", "csv_path": str(split_path)},
                "patching": {"mask_threshold": 127},
                "inference": {"threshold": 0.5},
            }

            def predictor(*args):
                path = args[1]
                probabilities = np.zeros((4, 5), dtype=np.float32)
                if Path(path).stem == "train":
                    probabilities[:2, :2] = 0.8
                return probabilities

            output = root / "val-train-set-evaluation"
            result = run_val_train_set_evaluation(
                root / "fold_0" / "best.pt", config, output, torch.device("cpu"),
                model=torch.nn.Identity(), predictor=predictor,
            )

            self.assertEqual(result["num_train_images"], 1)
            self.assertEqual(result["num_validation_images"], 1)
            self.assertTrue((output / "overlays" / "train" / "train_overlay.png").is_file())
            self.assertTrue((output / "overlays" / "validation" / "val_overlay.png").is_file())
            self.assertFalse((output / "masks").exists())
            self.assertFalse((output / "probabilities").exists())
            with (output / "val_train_set_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["split"] for row in rows],
                ["train", "validation", "train_mean", "validation_mean", "train_validation_mean"],
            )
            self.assertAlmostEqual(float(rows[0]["dice"]), 1.0)
            self.assertAlmostEqual(float(rows[1]["dice"]), 1.0)
            self.assertAlmostEqual(float(rows[-1]["dice"]), 1.0)


if __name__ == "__main__":
    unittest.main()
