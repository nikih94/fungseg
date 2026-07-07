from __future__ import annotations

import unittest

from src.data.folds import make_grouped_kfold_splits
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


if __name__ == "__main__":
    unittest.main()
