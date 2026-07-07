from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.qualitative_evaluation import (
    CheckpointEntry,
    SelectedCrop,
    _cross_fold_checkpoint_entries,
    _is_kfold_run,
    discover_manifest_checkpoints,
    intersecting_patch_coordinates,
    metric_row,
    select_qualitative_crop,
)


class QualitativeEvaluationTests(unittest.TestCase):
    def test_selects_mixed_foreground_crop(self) -> None:
        mask = np.zeros((768, 768), dtype=np.uint8)
        mask[180:260, 180:260] = 255

        crop = select_qualitative_crop(
            mask_array=mask,
            patch_size=256,
            stride=128,
            crop_patch_grid=(3, 3),
            mask_threshold=127,
            min_foreground_ratio=0.005,
            max_foreground_ratio=0.15,
        )

        self.assertEqual((crop.width, crop.height), (512, 512))
        self.assertGreaterEqual(crop.foreground_ratio, 0.005)
        self.assertLessEqual(crop.foreground_ratio, 0.15)
        self.assertEqual(crop.selection_reason, "in_range")

    def test_seeded_selection_is_repeatable(self) -> None:
        mask = np.zeros((100, 600), dtype=np.uint8)
        for index in range(6):
            mask[:10, index * 100 : (index + 1) * 100] = 255

        first = select_qualitative_crop(
            mask_array=mask,
            patch_size=100,
            stride=100,
            crop_patch_grid=(1, 1),
            mask_threshold=127,
            min_foreground_ratio=0.005,
            max_foreground_ratio=0.15,
            rng=np.random.default_rng(1),
        )
        repeat = select_qualitative_crop(
            mask_array=mask,
            patch_size=100,
            stride=100,
            crop_patch_grid=(1, 1),
            mask_threshold=127,
            min_foreground_ratio=0.005,
            max_foreground_ratio=0.15,
            rng=np.random.default_rng(1),
        )
        different_seed = select_qualitative_crop(
            mask_array=mask,
            patch_size=100,
            stride=100,
            crop_patch_grid=(1, 1),
            mask_threshold=127,
            min_foreground_ratio=0.005,
            max_foreground_ratio=0.15,
            rng=np.random.default_rng(2),
        )

        self.assertEqual(first, repeat)
        self.assertNotEqual((first.x, first.y), (different_seed.x, different_seed.y))

    def test_border_patches_are_included_for_crop_stitching(self) -> None:
        crop = SelectedCrop(
            x=128,
            y=128,
            width=256,
            height=256,
            foreground_ratio=0.01,
            selection_reason="in_range",
        )

        coordinates = intersecting_patch_coordinates(
            image_width=512,
            image_height=512,
            patch_size=256,
            stride=128,
            crop=crop,
        )

        self.assertEqual(len(coordinates), 9)
        self.assertIn((0, 0), coordinates)
        self.assertIn((256, 256), coordinates)

    def test_discovers_manifest_checkpoints_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            fold_dir = run_dir / "fold_0"
            fold_dir.mkdir()
            for name in ["best_epochs_001_010.pt", "best.pt", "last.pt"]:
                (fold_dir / name).write_bytes(b"checkpoint")
            (fold_dir / "checkpoint_manifest.csv").write_text(
                "\n".join(
                    [
                        "checkpoint,path,reason,epoch,epoch_start,epoch_end,monitor,monitor_value",
                        f"best_epochs_001_010.pt,runs/{run_dir.name}/fold_0/best_epochs_001_010.pt,interval_best,8,1,10,val_dice,0.8",
                        "best.pt,best.pt,global_best,8,8,8,val_dice,0.8",
                        "last.pt,last.pt,last,10,10,10,val_dice,0.7",
                    ]
                ),
                encoding="utf-8",
            )

            entries = discover_manifest_checkpoints(run_dir)

        self.assertEqual([entry.checkpoint for entry in entries], ["best_epochs_001_010.pt", "best.pt", "last.pt"])
        self.assertEqual(entries[0].fold, 0)
        self.assertEqual(entries[0].epoch_start, 1)
        self.assertAlmostEqual(entries[0].monitor_value or 0.0, 0.8)

    def test_metric_row_contains_crop_metrics(self) -> None:
        entry = CheckpointEntry(
            fold=0,
            checkpoint="best.pt",
            path=Path("fold_0/best.pt"),
            reason="global_best",
            epoch=4,
            epoch_start=4,
            epoch_end=4,
            monitor="val_dice",
            monitor_value=0.9,
        )
        crop = SelectedCrop(0, 0, 4, 4, 0.25, "in_range")
        target = np.zeros((4, 4), dtype=np.uint8)
        target[:2, :2] = 1
        prediction = target.copy()

        row = metric_row(Path("image.tif"), entry, crop, prediction, target)

        self.assertEqual(row["image"], "image.tif")
        self.assertEqual(row["checkpoint"], "best.pt")
        self.assertEqual(row["crop_width"], 4)
        self.assertAlmostEqual(row["dice"], 1.0)
        self.assertAlmostEqual(row["iou"], 1.0)

    def test_cross_fold_entries_prefer_global_best_per_fold(self) -> None:
        entries = [
            CheckpointEntry(0, "last.pt", Path("fold_0/last.pt"), "last", 10, 10, 10, "val_dice", 0.6),
            CheckpointEntry(0, "best.pt", Path("fold_0/best.pt"), "global_best", 8, 8, 8, "val_dice", 0.8),
            CheckpointEntry(1, "last.pt", Path("fold_1/last.pt"), "last", 10, 10, 10, "val_dice", 0.7),
            CheckpointEntry(1, "best.pt", Path("fold_1/best.pt"), "global_best", 7, 7, 7, "val_dice", 0.9),
        ]

        selected = _cross_fold_checkpoint_entries(entries)

        self.assertEqual([(entry.fold, entry.checkpoint) for entry in selected], [(0, "best.pt"), (1, "best.pt")])

    def test_cross_fold_entries_fall_back_to_best_named_checkpoint(self) -> None:
        entries = [
            CheckpointEntry(0, "last.pt", Path("fold_0/last.pt"), "last", 10, 10, 10, "val_dice", 0.6),
            CheckpointEntry(0, "best.pt", Path("fold_0/best.pt"), "manual", 8, 8, 8, "val_dice", 0.8),
            CheckpointEntry(1, "last.pt", Path("fold_1/last.pt"), "last", 10, 10, 10, "val_dice", 0.7),
        ]

        selected = _cross_fold_checkpoint_entries(entries)

        self.assertEqual([(entry.fold, entry.checkpoint) for entry in selected], [(0, "best.pt"), (1, "last.pt")])

    def test_kfold_detection_uses_saved_split_mode(self) -> None:
        self.assertTrue(_is_kfold_run({"split": {"mode": "kfold"}}))
        self.assertFalse(_is_kfold_run({"split": {"mode": "train_val"}}))
        self.assertFalse(_is_kfold_run({}))


if __name__ == "__main__":
    unittest.main()
