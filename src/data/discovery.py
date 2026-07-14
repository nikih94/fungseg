from __future__ import annotations

from pathlib import Path

from PIL import Image


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



def discover_image_mask_sets(
    images_dir: str | Path,
    mask_dirs: dict[str, str | Path],
    image_extensions: list[str] | None = None,
) -> tuple[list[tuple[Path, dict[str, Path]]], dict[str, object]]:
    """Discover complete image/mask sets and validate their dimensions."""
    images_root = Path(images_dir)
    extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    normalized_extensions = {ext.lower() for ext in extensions}
    image_files = [
        path for path in sorted(images_root.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    image_map = {path.stem: path for path in image_files}
    mask_maps = {
        name: {path.stem: path for path in sorted(Path(directory).iterdir()) if path.is_file()}
        for name, directory in mask_dirs.items()
    }
    missing_masks = {
        name: sorted(set(image_map) - set(mask_map)) for name, mask_map in mask_maps.items()
    }
    missing_images = {
        name: sorted(set(mask_map) - set(image_map)) for name, mask_map in mask_maps.items()
    }
    complete_stems = set(image_map)
    for mask_map in mask_maps.values():
        complete_stems &= set(mask_map)

    sets: list[tuple[Path, dict[str, Path]]] = []
    dimension_mismatches: list[dict[str, object]] = []
    for stem in sorted(complete_stems):
        image_path = image_map[stem]
        with Image.open(image_path) as image:
            image_size = image.size
        named_paths = {name: mask_map[stem] for name, mask_map in mask_maps.items()}
        mismatched = False
        for name, mask_path in named_paths.items():
            with Image.open(mask_path) as mask:
                mask_size = mask.size
            if mask_size != image_size:
                mismatched = True
                dimension_mismatches.append({
                    "stem": stem, "class": name, "image_size": list(image_size),
                    "mask_size": list(mask_size),
                })
        if not mismatched:
            sets.append((image_path, named_paths))

    return sets, {
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "dimension_mismatches": dimension_mismatches,
    }
