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
    optional_mask_dirs: dict[str, str | Path] | None = None,
) -> tuple[list[tuple[Path, dict[str, Path]]], dict[str, object]]:
    """Discover complete required mask sets plus any valid optional masks."""
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
    optional_mask_maps = {
        name: {path.stem: path for path in sorted(Path(directory).iterdir()) if path.is_file()}
        for name, directory in (optional_mask_dirs or {}).items()
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
    optional_dimension_mismatches: list[dict[str, object]] = []
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
            for name, mask_map in optional_mask_maps.items():
                mask_path = mask_map.get(stem)
                if mask_path is None:
                    continue
                with Image.open(mask_path) as mask:
                    mask_size = mask.size
                if mask_size != image_size:
                    optional_dimension_mismatches.append({
                        "stem": stem, "class": name, "image_size": list(image_size),
                        "mask_size": list(mask_size),
                    })
                    continue
                named_paths[name] = mask_path
            sets.append((image_path, named_paths))

    return sets, {
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "dimension_mismatches": dimension_mismatches,
        "optional_masks_without_images": {
            name: sorted(set(mask_map) - set(image_map))
            for name, mask_map in optional_mask_maps.items()
        },
        "optional_dimension_mismatches": optional_dimension_mismatches,
    }


def discovery_diagnostic_messages(diagnostics: dict[str, object]) -> list[str]:
    """Describe incomplete or invalid image/mask sets using their source stems."""
    messages: list[str] = []

    missing_masks = diagnostics.get("missing_masks", [])
    if isinstance(missing_masks, dict):
        for class_name, stems in sorted(missing_masks.items()):
            if stems:
                messages.append(
                    f"missing {class_name} masks for: "
                    + ", ".join(sorted(str(stem) for stem in stems))
                )
    elif missing_masks:
        messages.append(
            "missing masks for: " + ", ".join(sorted(str(stem) for stem in missing_masks))
        )

    missing_images = diagnostics.get("missing_images", [])
    if isinstance(missing_images, dict):
        for class_name, stems in sorted(missing_images.items()):
            if stems:
                messages.append(
                    f"{class_name} masks without images for: "
                    + ", ".join(sorted(str(stem) for stem in stems))
                )
    elif missing_images:
        messages.append(
            "masks without images for: "
            + ", ".join(sorted(str(stem) for stem in missing_images))
        )

    dimension_mismatches = diagnostics.get("dimension_mismatches", [])
    if dimension_mismatches:
        mismatch_names = sorted(
            f"{item.get('stem')} ({item.get('class')})"
            for item in dimension_mismatches
            if isinstance(item, dict)
        )
        if mismatch_names:
            messages.append("dimension mismatches for: " + ", ".join(mismatch_names))

    optional_without_images = diagnostics.get("optional_masks_without_images", {})
    if isinstance(optional_without_images, dict):
        for mask_name, stems in sorted(optional_without_images.items()):
            if stems:
                messages.append(
                    f"optional {mask_name} masks without images for: "
                    + ", ".join(sorted(str(stem) for stem in stems))
                )

    optional_mismatches = diagnostics.get("optional_dimension_mismatches", [])
    if optional_mismatches:
        mismatch_names = sorted(
            f"{item.get('stem')} ({item.get('class')})"
            for item in optional_mismatches
            if isinstance(item, dict)
        )
        if mismatch_names:
            messages.append(
                "ignored optional dimension mismatches for: " + ", ".join(mismatch_names)
            )

    return messages
