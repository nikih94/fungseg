from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.data.folds import SplitDefinition, make_csv_train_val_test_split, make_grouped_kfold_splits
from src.train import split_manifest_rows


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
