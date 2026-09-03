from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def contiguous_completed_folds(rows: list[dict[str, Any]], total_folds: int) -> list[int]:
    present = {int(row["fold"]) for row in rows if str(row.get("fold", "")).strip()}
    completed: list[int] = []
    for fold in range(total_folds):
        if fold not in present:
            break
        completed.append(fold)
    return completed


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_completed_fold(
    run_dir: Path,
    fold: int,
    test_required: bool,
) -> None:
    fold_dir = run_dir / f"fold_{fold}"
    required = [fold_dir / "metrics.json", fold_dir / "best_current.pt", fold_dir / "best_val_loss.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if test_required:
        comparison = run_dir / "test-evaluation" / "checkpoint_comparison.csv"
        rows = read_csv_rows(comparison)
        if not any(int(row.get("fold", -1)) == fold for row in rows):
            missing.append(f"test comparison row for fold {fold}")
    if missing:
        raise RuntimeError(
            f"Completed fold {fold} is missing required artifacts: " + ", ".join(missing)
        )


def clean_incomplete_folds(run_dir: Path, first_incomplete: int, total_folds: int) -> list[str]:
    removed: list[str] = []
    for fold in range(first_incomplete, total_folds):
        for path in (
            run_dir / f"fold_{fold}",
            run_dir / "test-evaluation" / f"fold_{fold}",
        ):
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(run_dir)))
    return removed


def append_resume_history(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "resume_history.json"
    history = []
    if path.is_file():
        history = json.loads(path.read_text(encoding="utf-8")).get("events", [])
    history.append({"resumed_at": datetime.now().astimezone().isoformat(), **payload})
    atomic_json(path, {"events": history})
