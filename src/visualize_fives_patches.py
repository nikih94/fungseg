from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from src.data.fives import build_fives_patch_records, discover_fives_pairs
from src.patching import PatchRecord, crop_and_pad_array
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the four centered FIVES training patches.")
    parser.add_argument("--config", default="config.yaml", help="Config supplying patch size and output root.")
    parser.add_argument("--image", default=None, help="Optional FIVES image path or filename stem.")
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    return parser.parse_args()


def select_fives_pair(
    pairs: list[tuple[Path, Path]],
    requested_image: str | None,
) -> tuple[Path, Path]:
    if requested_image is None:
        return pairs[0]
    requested = Path(requested_image)
    for image_path, mask_path in pairs:
        if image_path.stem == requested.stem or image_path == requested:
            return image_path, mask_path
    raise ValueError(f"FIVES image was not found: {requested_image}")


def create_fives_patch_visualization(
    image_path: Path,
    mask_path: Path,
    records: list[PatchRecord],
    output_path: Path,
) -> Path:
    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"))
    with Image.open(mask_path) as mask:
        mask_array = np.array(mask.convert("L"))

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    source_axis = figure.add_subplot(grid[:, 0])
    source_axis.imshow(image_array)
    source_axis.set_title(f"{image_path.name}: centered 2×2 training area")
    colors = ("#ff3b30", "#ff9500", "#34c759", "#007aff")
    for number, (record, color) in enumerate(zip(records, colors), start=1):
        source_axis.add_patch(Rectangle(
            (record.x, record.y), record.patch_size, record.patch_size,
            fill=False, edgecolor=color, linewidth=3,
        ))
        source_axis.text(
            record.x + 12, record.y + 36, str(number), color=color, fontsize=14, weight="bold",
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
        )
    source_axis.axis("off")

    for index, (record, color) in enumerate(zip(records, colors)):
        axis = figure.add_subplot(grid[index // 2, 1 + (index % 2)])
        image_patch = crop_and_pad_array(image_array, record.x, record.y, record.patch_size)
        mask_patch = crop_and_pad_array(mask_array, record.x, record.y, record.patch_size)
        overlay = image_patch.copy()
        vessel_pixels = mask_patch > 127
        overlay[vessel_pixels] = (
            0.5 * overlay[vessel_pixels] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        axis.imshow(overlay)
        axis.set_title(f"Patch {index + 1}: x={record.x}, y={record.y}", color=color)
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pairs = discover_fives_pairs(config["data"]["image_extensions"])
    image_path, mask_path = select_fives_pair(pairs, args.image)
    records = build_fives_patch_records(
        [(image_path, mask_path)], int(config["patching"]["patch_size"])
    )
    output_path = Path(args.output) if args.output else (
        Path(config["paths"]["outputs_dir"])
        / config["project"]["name"]
        / "fives_patch_visualization"
        / f"{image_path.stem}_fives_center_patches.png"
    )
    saved_path = create_fives_patch_visualization(image_path, mask_path, records, output_path)
    print(f"Saved FIVES patch visualization to {saved_path}")


if __name__ == "__main__":
    main()
