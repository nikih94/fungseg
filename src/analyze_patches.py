from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.discovery import discover_image_mask_pairs
from src.patching import _compute_positions, build_original_image_records, crop_and_pad_array
from src.utils.config import load_config
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze patch foreground distribution from masks.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def collect_foreground_counts(
    original_records,
    patch_size: int,
    stride: int,
    mask_threshold: int,
) -> list[int]:
    foreground_counts: list[int] = []

    for record in original_records:
        xs = _compute_positions(record.width, patch_size, stride)
        ys = _compute_positions(record.height, patch_size, stride)

        with Image.open(record.mask_path) as mask_image:
            mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)

        for y in ys:
            for x in xs:
                mask_patch = crop_and_pad_array(mask_array, x, y, patch_size)
                foreground_counts.append(int((mask_patch > mask_threshold).sum()))

    return foreground_counts


def save_histogram(
    foreground_counts: list[int],
    output_path: Path,
    min_foreground_pixels: int,
    max_display_foreground_pixels: int = 1000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overflow_edge = max_display_foreground_pixels + 1
    histogram_values = [min(count, overflow_edge) for count in foreground_counts]
    bin_width = 25
    bin_edges = list(range(0, max_display_foreground_pixels + bin_width, bin_width))
    if bin_edges[-1] != max_display_foreground_pixels:
        bin_edges.append(max_display_foreground_pixels)
    bin_edges.append(overflow_edge)

    plt.figure(figsize=(10, 6))
    plt.hist(histogram_values, bins=bin_edges, color="#2e7d32", edgecolor="black")
    plt.axvline(min_foreground_pixels, color="red", linestyle="--", linewidth=2, label=f"Cutoff = {min_foreground_pixels}")
    plt.title("Foreground Pixel Distribution Per Patch")
    plt.xlabel("Foreground Pixels")
    plt.ylabel("Number of Patches")
    plt.xlim(0, overflow_edge + bin_width)
    plt.xticks(
        [0, 100, 250, 500, 750, 1000, overflow_edge],
        ["0", "100", "250", "500", "750", "1000", f">{max_display_foreground_pixels}"],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pairs, diagnostics = discover_image_mask_pairs(
        config["paths"]["images_dir"],
        config["paths"]["masks_dir"],
        config["data"]["image_extensions"],
    )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found.")

    original_records = build_original_image_records(pairs)
    data_cfg = config["data"]
    project_name = config["project"]["name"]

    foreground_counts = collect_foreground_counts(
        original_records=original_records,
        patch_size=int(data_cfg["patch_size"]),
        stride=int(data_cfg["stride"]),
        mask_threshold=int(data_cfg["mask_threshold"]),
    )
    if not foreground_counts:
        raise RuntimeError("No patches were generated from the discovered image/mask pairs.")

    total_patches = len(foreground_counts)
    min_foreground_pixels = int(data_cfg["min_foreground_pixels"])
    kept_patches = sum(count >= min_foreground_pixels for count in foreground_counts)
    discarded_patches = total_patches - kept_patches
    empty_patches = sum(count == 0 for count in foreground_counts)

    outputs_dir = ensure_dir(Path(config["paths"]["outputs_dir"]) / project_name)
    histogram_path = outputs_dir / "for_pix_distribution.png"
    save_histogram(foreground_counts, histogram_path, min_foreground_pixels)

    print(f"Images matched: {len(original_records)}")
    print(f"Total candidate patches: {total_patches}")
    print(f"Patches kept (foreground >= {min_foreground_pixels}): {kept_patches}")
    print(f"Patches discarded (foreground < {min_foreground_pixels}): {discarded_patches}")
    print(f"Strictly empty patches (0 foreground pixels): {empty_patches}")
    print(
        f"Foreground pixel stats: min={min(foreground_counts)} max={max(foreground_counts)} mean={np.mean(foreground_counts):.2f}"
    )
    print(f"Histogram saved to: {histogram_path}")

    if diagnostics["missing_masks"]:
        print(f"Warning: missing masks for {len(diagnostics['missing_masks'])} images")
    if diagnostics["missing_images"]:
        print(f"Warning: found {len(diagnostics['missing_images'])} masks without matching images")


if __name__ == "__main__":
    main()
