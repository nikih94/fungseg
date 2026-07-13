from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitDefinition:
    train_sources: list[str]
    val_sources: list[str]
    test_sources: list[str]


def make_grouped_kfold_splits(
    source_ids: Iterable[str],
    n_splits: int,
    shuffle_groups: bool,
    random_state: int | None,
) -> list[tuple[list[str], list[str]]]:
    unique_sources = list(dict.fromkeys(source_ids))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(unique_sources) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is larger than the number of unique source images={len(unique_sources)}."
        )

    if shuffle_groups:
        rng = random.Random(random_state)
        rng.shuffle(unique_sources)

    folds: dict[int, list[str]] = defaultdict(list)
    for index, source_id in enumerate(unique_sources):
        folds[index % n_splits].append(source_id)

    splits: list[tuple[list[str], list[str]]] = []
    for fold_idx in range(n_splits):
        val_sources = folds[fold_idx]
        train_sources = [
            source_id
            for other_fold_idx, values in folds.items()
            if other_fold_idx != fold_idx
            for source_id in values
        ]
        splits.append((train_sources, val_sources))
    return splits


def _source_lookup(source_ids: Iterable[str]) -> dict[str, str]:
    unique_sources = list(dict.fromkeys(source_ids))
    lookup = {source_id: source_id for source_id in unique_sources}
    lookup.update({Path(source_id).name: source_id for source_id in unique_sources})
    lookup.update({Path(source_id).stem: source_id for source_id in unique_sources})
    return lookup


def _normalize_split_label(label: str) -> str:
    normalized = label.strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported CSV split label '{label}'. Expected train, validation/val, or test."
        )
    return aliases[normalized]


def make_csv_train_val_test_split(
    source_ids: Iterable[str],
    csv_path: str | Path,
) -> SplitDefinition:
    unique_sources = list(dict.fromkeys(source_ids))
    lookup = _source_lookup(unique_sources)
    assignments: dict[str, str] = {}
    csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "filename" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"Split CSV must contain 'filename' and 'split' columns: {csv_path}")

        for row_index, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            raw_split = (row.get("split") or "").strip()
            if not filename or not raw_split:
                raise ValueError(f"Split CSV row {row_index} must include filename and split.")
            if filename not in lookup and Path(filename).name not in lookup and Path(filename).stem not in lookup:
                raise ValueError(f"Split CSV references unknown image: {filename}")

            source_id = lookup.get(filename) or lookup.get(Path(filename).name) or lookup[Path(filename).stem]
            split_label = _normalize_split_label(raw_split)
            previous = assignments.get(source_id)
            if previous is not None and previous != split_label:
                raise ValueError(
                    f"Split CSV assigns image '{source_id}' to multiple splits: {previous}, {split_label}."
                )
            assignments[source_id] = split_label

    unassigned_sources = [source_id for source_id in unique_sources if source_id not in assignments]
    if unassigned_sources:
        raise ValueError(
            "Split CSV does not assign every discovered image. Missing: "
            + ", ".join(sorted(unassigned_sources))
        )

    train_sources = [source_id for source_id in unique_sources if assignments[source_id] == "train"]
    val_sources = [source_id for source_id in unique_sources if assignments[source_id] == "val"]
    test_sources = [source_id for source_id in unique_sources if assignments[source_id] == "test"]

    if not train_sources:
        raise ValueError("CSV split requires at least one training image.")
    if not val_sources:
        raise ValueError("CSV split requires at least one validation image.")
    if not test_sources:
        raise ValueError("CSV split requires at least one test image.")

    return SplitDefinition(
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources,
    )


def make_manual_train_val_split(
    source_ids: Iterable[str],
    val_source_ids: Iterable[str],
) -> list[tuple[list[str], list[str]]]:
    unique_sources = list(dict.fromkeys(source_ids))
    source_set = set(unique_sources)
    requested_val_sources = list(dict.fromkeys(val_source_ids))

    if not requested_val_sources:
        raise ValueError("Manual train/val split requires at least one validation source_id.")

    normalized_lookup = {source_id: source_id for source_id in unique_sources}
    normalized_lookup.update({Path(source_id).stem: source_id for source_id in unique_sources})

    missing_sources = [source_id for source_id in requested_val_sources if source_id not in normalized_lookup]
    if missing_sources:
        raise ValueError(
            "Validation source_ids were not found in the discovered dataset: "
            + ", ".join(sorted(missing_sources))
        )

    resolved_val_sources = {normalized_lookup[source_id] for source_id in requested_val_sources}
    val_sources = [source_id for source_id in unique_sources if source_id in resolved_val_sources]
    train_sources = [source_id for source_id in unique_sources if source_id not in resolved_val_sources]

    if not train_sources:
        raise ValueError("Manual train/val split left no training images. Remove at least one validation source_id.")

    return [(train_sources, val_sources)]
