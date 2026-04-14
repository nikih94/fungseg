from __future__ import annotations

from pathlib import Path


def discover_image_mask_pairs(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_extensions: list[str],
) -> tuple[list[tuple[Path, Path]], dict[str, list[str]]]:
    images_root = Path(images_dir)
    masks_root = Path(masks_dir)

    normalized_extensions = {ext.lower() for ext in image_extensions}
    image_files = [
        path
        for path in sorted(images_root.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    mask_files = [path for path in sorted(masks_root.iterdir()) if path.is_file()]

    image_map = {path.stem: path for path in image_files}
    mask_map = {path.stem: path for path in mask_files}

    matched_stems = sorted(set(image_map) & set(mask_map))
    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))

    pairs = [(image_map[stem], mask_map[stem]) for stem in matched_stems]
    diagnostics = {
        "missing_masks": missing_masks,
        "missing_images": missing_images,
    }
    return pairs, diagnostics

