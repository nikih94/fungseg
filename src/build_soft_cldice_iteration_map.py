from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.analyze_soft_skeleton_iterations import discover_masks
from src.data.soft_cldice_iterations import required_soft_skeleton_iterations
from src.utils.io import save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the exact required soft-skeleton iterations per loci mask."
    )
    parser.add_argument("--mask-dir", default="data/loci_masks")
    parser.add_argument(
        "--output-csv", default="data/loci_soft_cldice_required_iterations.csv"
    )
    parser.add_argument("--threshold", type=int, default=127)
    return parser.parse_args()


def build_iteration_rows(mask_dir: str | Path, threshold: int = 127) -> list[dict]:
    rows: list[dict] = []
    for path in discover_masks(mask_dir):
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > threshold
        rows.append(
            {
                "mask_filename": path.name,
                "mask_stem": path.stem,
                "width": mask.shape[1],
                "height": mask.shape[0],
                "foreground_pixels": int(mask.sum()),
                "required_iterations": required_soft_skeleton_iterations(mask),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = build_iteration_rows(args.mask_dir, threshold=args.threshold)
    output_path = Path(args.output_csv)
    save_csv(output_path, rows)
    print(f"Wrote {len(rows)} loci-mask iteration rows to {output_path}")


if __name__ == "__main__":
    main()
