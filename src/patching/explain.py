from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.discovery import discover_image_mask_pairs
from src.data.folds import SplitDefinition, make_csv_train_val_test_split
from src.patching.core import (
    OriginalImageRecord,
    PatchRecord,
    _centered_source_bounds,
    build_original_image_records,
    build_patch_records,
)
from src.utils.config import load_config, resolve_mask_dir
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain dynamic patching for a config.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--target", default=None, help="Optional segmentation target override, e.g. loci or inoculum.")
    parser.add_argument("--image", default=None, help="Optional image path to draw a patch overlay for.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number used for deterministic training patching.")
    return parser.parse_args()


def _candidate_config(patching_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(patching_config)
    config["filter_empty_patches"] = False
    config["min_foreground_pixels"] = 0
    return config


def _scale_stats(records: list[PatchRecord]) -> dict[str, float]:
    scales = np.asarray([record.scale for record in records], dtype=np.float32)
    if scales.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "min": float(scales.min()),
        "max": float(scales.max()),
        "mean": float(scales.mean()),
        "p50": float(np.percentile(scales, 50)),
        "p90": float(np.percentile(scales, 90)),
        "p95": float(np.percentile(scales, 95)),
    }


def _resolution_bin_edges(patch_size: int, num_scaled_bins: int = 10) -> list[int]:
    return [
        int(round(patch_size * (1.0 + (index / num_scaled_bins))))
        for index in range(num_scaled_bins + 1)
    ]


def _resolution_bin_index(record: PatchRecord, bin_edges: list[int]) -> int:
    crop_size = int(record.source_crop_size or round(record.patch_size * record.scale))
    for index, edge in enumerate(bin_edges):
        if crop_size <= edge:
            return index
    return len(bin_edges) - 1


def _print_source_resolution_table(
    kept: list[PatchRecord],
    patch_size: int,
    title: str = "Patches by source and source-crop resolution:",
) -> None:
    bin_edges = _resolution_bin_edges(patch_size)
    source_rows: dict[str, list[int]] = {}
    for record in kept:
        row = source_rows.setdefault(record.source_id, [0 for _ in bin_edges])
        row[_resolution_bin_index(record, bin_edges)] += 1

    source_header = "source"
    total_header = "total"
    bin_headers = [f"<={edge}" for edge in bin_edges]
    bin_totals = [
        sum(row[index] for row in source_rows.values())
        for index in range(len(bin_edges))
    ]
    total_patches = sum(bin_totals)
    percentage_values = [
        f"{(100.0 * value / total_patches):.1f}%" if total_patches else "0.0%"
        for value in bin_totals
    ]
    source_width = max(
        len(source_header),
        len("percent"),
        *(len(source_id) for source_id in source_rows),
    ) if source_rows else len(source_header)
    total_width = max(
        len(total_header),
        len("100.0%"),
        *(len(str(sum(row))) for row in source_rows.values()),
    ) if source_rows else len(total_header)
    bin_widths = [
        max(
            len(header),
            len(percentage_values[index]),
            *(len(str(row[index])) for row in source_rows.values()),
        )
        for index, header in enumerate(bin_headers)
    ]

    print(title)
    header = (
        f"  {source_header:<{source_width}}  "
        f"{total_header:>{total_width}}  "
        + "  ".join(
            f"{header:>{width}}"
            for header, width in zip(bin_headers, bin_widths)
        )
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for source_id in sorted(source_rows):
        row = source_rows[source_id]
        print(
            f"  {source_id:<{source_width}}  "
            f"{sum(row):>{total_width}}  "
            + "  ".join(
                f"{value:>{width}}"
                for value, width in zip(row, bin_widths)
            )
        )
    print(
        f"  {'total':<{source_width}}  "
        f"{total_patches:>{total_width}}  "
        + "  ".join(
            f"{value:>{width}}"
            for value, width in zip(bin_totals, bin_widths)
        )
    )
    print(
        f"  {'percent':<{source_width}}  "
        f"{'100.0%':>{total_width}}  "
        + "  ".join(
            f"{value:>{width}}"
            for value, width in zip(percentage_values, bin_widths)
        )
    )


def _csv_split_definition(config: dict[str, Any], original_records: list[OriginalImageRecord]) -> SplitDefinition | None:
    split_cfg = config.get("split", {})
    if str(split_cfg.get("mode", "")).strip().lower() != "csv":
        return None
    csv_path = split_cfg.get("csv_path")
    if not csv_path:
        return None
    try:
        return make_csv_train_val_test_split(
            [record.source_id for record in original_records],
            csv_path=csv_path,
        )
    except (FileNotFoundError, ValueError):
        return None


def _print_split_resolution_tables(
    kept: list[PatchRecord],
    patch_size: int,
    split: SplitDefinition | None,
) -> None:
    if split is None:
        return

    split_sources = [
        ("train", set(split.train_sources)),
        ("validation", set(split.val_sources)),
        ("test", set(split.test_sources)),
    ]
    for split_name, source_ids in split_sources:
        split_records = [record for record in kept if record.source_id in source_ids]
        print()
        _print_source_resolution_table(
            split_records,
            patch_size,
            title=f"Patches by source and source-crop resolution ({split_name} split):",
        )


def _print_summary(
    original_records: list[OriginalImageRecord],
    candidates: list[PatchRecord],
    kept: list[PatchRecord],
    patching_config: dict[str, Any],
    epoch: int,
    seed: int,
    split: SplitDefinition | None = None,
) -> None:
    labels = Counter(record.scale_label for record in kept)
    discarded = max(0, len(candidates) - len(kept))
    stats = _scale_stats(kept)
    patch_size = int(patching_config["patch_size"])

    print(f"Images matched: {len(original_records)}")
    print(f"Epoch: {epoch}")
    print(f"RNG seed: {seed + epoch}")
    print(
        "Patch geometry: "
        f"patch_size={patching_config['patch_size']} "
        f"stride={patching_config['stride']} "
        f"overlap={patching_config['overlap']}"
    )
    print(f"Base candidate patches: {len(candidates)}")
    print(f"Patches kept: {len(kept)}")
    print(f"Patches discarded: {discarded}")
    print(f"Normal patches: {labels.get('normal', 0)}")
    print(f"Scaled-context patches: {labels.get('scaled_context', 0)}")
    print(
        "Scale stats: "
        f"min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} "
        f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} p95={stats['p95']:.4f}"
    )
    _print_source_resolution_table(kept, patch_size)
    _print_split_resolution_tables(kept, patch_size, split)


def _match_image_record(records: list[OriginalImageRecord], image_path: str) -> OriginalImageRecord:
    requested = Path(image_path).expanduser().resolve()
    for record in records:
        if record.image_path.expanduser().resolve() == requested:
            return record
    for record in records:
        if record.image_path.name == Path(image_path).name:
            return record
    raise ValueError(f"Image was not found among discovered image/mask pairs: {image_path}")


def _draw_overlay(
    record: OriginalImageRecord,
    records: list[PatchRecord],
    output_path: Path,
    epoch: int,
    seed: int,
) -> None:
    image = Image.open(record.image_path).convert("RGB")
    scaled_count = sum(1 for item in records if item.scale_label == "scaled_context")
    scaled_percent = (100.0 * scaled_count / len(records)) if records else 0.0
    header_height = 92
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{record.source_id} | epoch={epoch} seed={seed + epoch} | "
        f"scaled={scaled_count}/{len(records)} ({scaled_percent:.1f}%)"
    )
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)
    draw.rectangle((10, 36, 30, 56), outline=(44, 160, 44), width=3)
    draw.text((38, 39), "normal patch footprint", fill=(0, 0, 0), font=font)
    draw.rectangle((230, 36, 250, 56), outline=(255, 127, 14), width=3)
    draw.text((258, 39), "scaled patch footprint", fill=(0, 0, 0), font=font)
    draw.rectangle((450, 36, 470, 56), outline=(214, 39, 40), width=3)
    draw.line((450, 46, 470, 46), fill=(214, 39, 40), width=2)
    draw.text((478, 39), "scaled source crop", fill=(0, 0, 0), font=font)

    for patch_record in records:
        color = (44, 160, 44) if patch_record.scale_label == "normal" else (255, 127, 14)
        draw.rectangle(
            (
                patch_record.x,
                patch_record.y + header_height,
                patch_record.x + patch_record.patch_size,
                patch_record.y + patch_record.patch_size + header_height,
            ),
            outline=color,
            width=2,
        )
        if patch_record.scale_label == "scaled_context":
            x0, y0, x1, y1 = _centered_source_bounds(
                record.width,
                record.height,
                patch_record.x,
                patch_record.y,
                patch_record.patch_size,
                patch_record.scale,
            )
            draw.rectangle(
                (x0, y0 + header_height, x1, y1 + header_height),
                outline=(214, 39, 40),
                width=2,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.target is not None:
        config.setdefault("segmentation", {})["target"] = str(args.target)
    pairs, diagnostics = discover_image_mask_pairs(
        config["paths"]["images_dir"],
        resolve_mask_dir(config),
        config["data"]["image_extensions"],
    )
    if not pairs:
        raise RuntimeError("No matched image/mask pairs were found.")

    original_records = build_original_image_records(pairs)
    split = _csv_split_definition(config, original_records)
    patching_config = config["patching"]
    seed = int(config["train"]["seed"])
    candidates = build_patch_records(
        original_records,
        _candidate_config(patching_config),
        phase="train",
        epoch=int(args.epoch),
        base_seed=seed,
    )
    kept = build_patch_records(
        original_records,
        patching_config,
        phase="train",
        epoch=int(args.epoch),
        base_seed=seed,
    )
    _print_summary(original_records, candidates, kept, patching_config, int(args.epoch), seed, split=split)

    if args.image:
        target = str(config.get("segmentation", {}).get("target", "legacy"))
        image_record = _match_image_record(original_records, args.image)
        image_records = [record for record in kept if record.source_id == image_record.source_id]
        output_dir = ensure_dir(Path(config["paths"]["outputs_dir"]) / config["project"]["name"] / "patching_explain")
        output_path = output_dir / f"{Path(image_record.source_id).stem}_{target}_epoch_{int(args.epoch):03d}_patches.png"
        _draw_overlay(image_record, image_records, output_path, int(args.epoch), seed)
        print(f"Overlay saved to: {output_path}")

    if diagnostics["missing_masks"]:
        print(f"Warning: missing masks for {len(diagnostics['missing_masks'])} images")
    if diagnostics["missing_images"]:
        print(f"Warning: found {len(diagnostics['missing_images'])} masks without matching images")


if __name__ == "__main__":
    main()
