from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_cdt

from src.patching import OriginalImageRecord


def required_soft_skeleton_iterations(mask: np.ndarray) -> int:
    """Return the last useful soft-skeleton erosion for a crisp binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if not bool(binary.any()):
        return 0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    maximum_distance = int(distance_transform_cdt(padded, metric="taxicab").max())
    return max(0, maximum_distance - 1)


def load_training_iteration_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Soft-clDice iteration CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"mask_filename", "mask_stem", "training_iterations"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Soft-clDice iteration CSV {csv_path} is missing columns: {missing}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Soft-clDice iteration CSV is empty: {csv_path}")
    return rows


def map_training_iterations_to_sources(
    path: str | Path,
    records: Iterable[OriginalImageRecord],
) -> dict[str, int]:
    """Resolve adjusted mask rows to image source IDs through their mask paths."""
    rows = load_training_iteration_rows(path)
    by_filename: dict[str, int] = {}
    by_stem: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=2):
        filename = row["mask_filename"].strip()
        stem = row["mask_stem"].strip()
        if not filename or not stem:
            raise ValueError(f"Empty mask identifier at {path}:{line_number}.")
        try:
            iterations = int(row["training_iterations"])
        except ValueError as error:
            raise ValueError(
                f"Invalid training_iterations at {path}:{line_number}: "
                f"{row['training_iterations']!r}"
            ) from error
        if iterations < 0:
            raise ValueError(
                f"training_iterations must be non-negative at {path}:{line_number}."
            )
        if filename in by_filename or stem in by_stem:
            raise ValueError(
                f"Duplicate mask filename or stem at {path}:{line_number}: {filename}"
            )
        by_filename[filename] = iterations
        by_stem[stem] = iterations

    resolved: dict[str, int] = {}
    missing_sources: list[str] = []
    for record in records:
        loci_path = (
            record.mask_paths.get("loci")
            if record.mask_paths is not None
            else record.mask_path
        )
        if loci_path is None:
            missing_sources.append(record.source_id)
            continue
        value = by_filename.get(loci_path.name)
        if value is None:
            value = by_stem.get(loci_path.stem)
        if value is None:
            missing_sources.append(record.source_id)
        else:
            resolved[record.source_id] = value
    if missing_sources:
        raise ValueError(
            "Soft-clDice iteration CSV has no row for loci masks belonging to: "
            + ", ".join(sorted(missing_sources))
        )
    return resolved
