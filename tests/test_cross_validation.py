from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.data.folds import (
    SplitDefinition,
    make_csv_kfold_splits,
    make_csv_train_val_test_split,
    make_grouped_kfold_splits,
)
from src.train import (
    build_checkpoint_test_comparison,
    build_cross_fold_test_summary,
    checkpoint_selection_from_history,
    persist_checkpoint_test_comparison,
    persist_cross_fold_test_summary,
    split_manifest_rows,
)


class CrossValidationTests(unittest.TestCase):
    def test_grouped_kfold_uses_each_source_once_as_validation(self) -> None:
        source_ids = [f"source_{index}.tif" for index in range(7)]
        splits = make_grouped_kfold_splits(
            source_ids,
            n_splits=3,
            shuffle_groups=False,
            random_state=42,
        )

        validation_sources = [source_id for _, val_sources in splits for source_id in val_sources]
        self.assertCountEqual(validation_sources, source_ids)
        for train_sources, val_sources in splits:
            self.assertFalse(set(train_sources) & set(val_sources))

    def test_grouped_kfold_errors_when_too_many_splits(self) -> None:
        with self.assertRaises(ValueError):
            make_grouped_kfold_splits(
                ["a.tif", "b.tif"],
                n_splits=3,
                shuffle_groups=False,
                random_state=None,
            )

    def test_csv_kfold_keeps_test_fixed_and_covers_train_validation_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            cv_sources = [f"source_{index}.tif" for index in range(8)]
            test_sources = ["test_a.tif", "test_b.tif"]
            csv_path.write_text(
                "filename,split\n"
                + "\n".join(
                    [
                        *[f"{source},train" for source in cv_sources[:6]],
                        *[f"{source},validation" for source in cv_sources[6:]],
                        *[f"{source},test" for source in test_sources],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            first = make_csv_kfold_splits(
                cv_sources + test_sources,
                csv_path,
                n_splits=3,
                shuffle_groups=True,
                random_state=42,
            )
            second = make_csv_kfold_splits(
                cv_sources + test_sources,
                csv_path,
                n_splits=3,
                shuffle_groups=True,
                random_state=42,
            )

        self.assertEqual(first, second)
        self.assertEqual([len(split.val_sources) for split in first], [3, 3, 2])
        self.assertCountEqual(
            [source for split in first for source in split.val_sources],
            cv_sources,
        )
        for split in first:
            self.assertEqual(split.test_sources, test_sources)
            self.assertFalse(set(split.train_sources) & set(split.val_sources))
            self.assertFalse(set(split.train_sources) & set(split.test_sources))
            self.assertFalse(set(split.val_sources) & set(split.test_sources))

    def test_cross_fold_test_summary_aggregates_all_available_metrics(self) -> None:
        fold_results = [
            {
                "fold": 0,
                "checkpoint": "fold_0/best.pt",
                "output_dir": "test-evaluation/fold_0",
                "num_test_images": 2,
                "threshold": "argmax",
                "mean_dice_loci": 0.8,
                "mean_dice_join": None,
            },
            {
                "fold": 1,
                "checkpoint": "fold_1/best.pt",
                "output_dir": "test-evaluation/fold_1",
                "num_test_images": 2,
                "threshold": "argmax",
                "mean_dice_loci": 0.6,
                "mean_dice_join": 0.4,
            },
        ]

        summary, fold_rows, metric_rows = build_cross_fold_test_summary(fold_results)

        self.assertEqual(summary["num_folds"], 2)
        self.assertEqual(summary["num_test_images"], 2)
        self.assertAlmostEqual(summary["metrics"]["mean_dice_loci"]["mean"], 0.7)
        self.assertAlmostEqual(summary["metrics"]["mean_dice_loci"]["std"], 0.1)
        self.assertEqual(summary["metrics"]["mean_dice_loci"]["num_folds"], 2)
        self.assertEqual(summary["metrics"]["mean_dice_join"]["num_folds"], 1)
        self.assertIsNone(fold_rows[0]["mean_dice_join"])
        self.assertEqual({row["metric"] for row in metric_rows}, {"mean_dice_join", "mean_dice_loci"})

        with tempfile.TemporaryDirectory() as tmpdir:
            persisted = persist_cross_fold_test_summary(tmpdir, fold_results)
            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "fold_metrics.csv").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
        self.assertEqual(persisted, summary)

    def test_checkpoint_test_comparison_is_side_by_side_and_cross_fold(self) -> None:
        results = []
        for fold, current_dice, loss_dice in ((0, 0.8, 0.7), (1, 0.6, 0.9)):
            for checkpoint_name, monitor, value in (
                ("best_current.pt", "val_dice_cldice_per_image", current_dice),
                ("best_val_loss.pt", "val_loss", loss_dice),
            ):
                results.append({
                    "fold": fold,
                    "checkpoint_name": checkpoint_name,
                    "selection_monitor": monitor,
                    "selection_mode": "min" if monitor == "val_loss" else "max",
                    "selection_epoch": fold + 2,
                    "selection_value": value,
                    "checkpoint": f"fold_{fold}/{checkpoint_name}",
                    "output_dir": f"test-evaluation/fold_{fold}/{Path(checkpoint_name).stem}",
                    "num_test_images": 2,
                    "num_join_images": 1,
                    "threshold": "argmax",
                    "mean_dice": value,
                    "mean_dice_join": None if fold == 0 else value - 0.1,
                })

        rows, summary_rows = build_checkpoint_test_comparison(results)

        self.assertEqual(len(rows), 4)
        summaries = {row["checkpoint_name"]: row for row in summary_rows}
        self.assertAlmostEqual(summaries["best_current.pt"]["mean_dice"], 0.7)
        self.assertAlmostEqual(summaries["best_current.pt"]["mean_dice_std"], 0.1)
        self.assertEqual(
            summaries["best_val_loss.pt"]["mean_dice_join_num_folds"],
            1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            persisted = persist_checkpoint_test_comparison(tmpdir, results, total_folds=2)
            output_dir = Path(tmpdir)
            self.assertEqual(persisted, summary_rows)
            self.assertTrue((output_dir / "checkpoint_comparison.csv").is_file())
            self.assertTrue((output_dir / "monitor_comparison_summary.csv").is_file())
            self.assertTrue((output_dir / "fold_0" / "checkpoint_comparison.csv").is_file())
            self.assertTrue((output_dir / "fold_1" / "checkpoint_comparison.csv").is_file())

    def test_checkpoint_selection_uses_monitor_direction_and_first_tie(self) -> None:
        history = [
            {"epoch": 1, "score": 0.4, "loss": 0.7},
            {"epoch": 2, "score": 0.8, "loss": 0.2},
            {"epoch": 3, "score": 0.8, "loss": 0.4},
        ]

        self.assertEqual(checkpoint_selection_from_history(history, "score", "max"), (2, 0.8))
        self.assertEqual(checkpoint_selection_from_history(history, "loss", "min"), (2, 0.2))

    def test_split_manifest_rows_include_train_and_val_sources(self) -> None:
        rows = split_manifest_rows(
            [
                (["a.tif", "b.tif"], ["c.tif"]),
                (["c.tif"], ["a.tif", "b.tif"]),
            ]
        )

        self.assertEqual(
            rows,
            [
                {"fold": 0, "split": "train", "source_id": "a.tif"},
                {"fold": 0, "split": "train", "source_id": "b.tif"},
                {"fold": 0, "split": "val", "source_id": "c.tif"},
                {"fold": 1, "split": "train", "source_id": "c.tif"},
                {"fold": 1, "split": "val", "source_id": "a.tif"},
                {"fold": 1, "split": "val", "source_id": "b.tif"},
            ],
        )

    def test_split_manifest_rows_include_test_sources(self) -> None:
        rows = split_manifest_rows(
            [
                SplitDefinition(
                    train_sources=["a.tif"],
                    val_sources=["b.tif"],
                    test_sources=["c.tif"],
                )
            ]
        )

        self.assertEqual(
            rows,
            [
                {"fold": 0, "split": "train", "source_id": "a.tif"},
                {"fold": 0, "split": "val", "source_id": "b.tif"},
                {"fold": 0, "split": "test", "source_id": "c.tif"},
            ],
        )

    def test_csv_split_parses_train_validation_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "filename,split",
                        "a.tif,train",
                        "b.tif,validation",
                        "c.tif,test",
                    ]
                ),
                encoding="utf-8",
            )

            split = make_csv_train_val_test_split(["a.tif", "b.tif", "c.tif"], csv_path)

        self.assertEqual(split.train_sources, ["a.tif"])
        self.assertEqual(split.val_sources, ["b.tif"])
        self.assertEqual(split.test_sources, ["c.tif"])

    def test_csv_split_ignores_future_rows_with_named_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            csv_path.write_text(
                "filename,split\na.tif,train\nb.tif,validation\nc.tif,test\nfuture.tif,train\n",
                encoding="utf-8",
            )

            with self.assertWarnsRegex(RuntimeWarning, "future.tif"):
                split = make_csv_train_val_test_split(["a.tif", "b.tif", "c.tif"], csv_path)

        self.assertEqual(split.train_sources, ["a.tif"])
        self.assertEqual(split.val_sources, ["b.tif"])
        self.assertEqual(split.test_sources, ["c.tif"])

    def test_csv_split_validates_future_row_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            csv_path.write_text(
                "filename,split\na.tif,train\nb.tif,validation\nc.tif,test\nfuture.tif,pending\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported CSV split label"):
                make_csv_train_val_test_split(["a.tif", "b.tif", "c.tif"], csv_path)

    def test_csv_split_rejects_conflicting_future_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            csv_path.write_text(
                (
                    "filename,split\na.tif,train\nb.tif,validation\nc.tif,test\n"
                    "future.tif,train\nfuture.png,test\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple splits"):
                make_csv_train_val_test_split(["a.tif", "b.tif", "c.tif"], csv_path)

    def test_csv_split_rejects_unassigned_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "image_splits.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "filename,split",
                        "a.tif,train",
                        "b.tif,validation",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                make_csv_train_val_test_split(["a.tif", "b.tif", "c.tif"], csv_path)


if __name__ == "__main__":
    unittest.main()
