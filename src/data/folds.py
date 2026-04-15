from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


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
