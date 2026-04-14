"""
convert_masks.py
────────────────
Converts multiple CVAT COCO-format exports into binary PNG masks.

Expected input structure:
    data/cvat-export/
        batch_1/
            annotations/instances_default.json
            images/default/*.jpg
        batch_2/
            annotations/instances_default.json
            images/default/*.jpg
        ...

Output:
    data/output/images/<subfolder>_<image_filename>
    data/output/masks/<subfolder>_<image_stem>.png

White = foreground
Black = background
"""

import json
import shutil
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ---------------------------------------------------------------------
# Hardcoded paths
# ---------------------------------------------------------------------
INPUT_ROOT = Path("data/cvat-export")
OUTPUT_ROOT = Path("data")

OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASKS_DIR = OUTPUT_ROOT / "masks"

IGNORE_CATEGORIES = ["inoculum"]


def draw_annotation(mask: Image.Image, ann: dict, width: int, height: int, fill: int) -> Image.Image:
    """
    Draw a single COCO annotation onto a PIL mask.
    fill=255 -> foreground
    fill=0   -> erase / ignore region
    """
    draw = ImageDraw.Draw(mask)
    seg = ann.get("segmentation", [])

    if isinstance(seg, list):
        # Polygon format: [[x1,y1,x2,y2,...], ...]
        for poly in seg:
            if len(poly) >= 6:
                coords = list(zip(poly[::2], poly[1::2]))
                draw.polygon(coords, fill=fill)

    elif isinstance(seg, dict) and seg.get("counts") is not None:
        # RLE format — requires pycocotools
        try:
            from pycocotools import mask as coco_mask

            if isinstance(seg.get("counts"), list):
                rle = coco_mask.frPyObjects(seg, height, width)
            else:
                rle = seg

            rle_mask = coco_mask.decode(rle)

            # decode may return HxW or HxWxN
            if rle_mask.ndim == 3:
                rle_mask = np.any(rle_mask, axis=2).astype(np.uint8)

            mask_arr = np.array(mask)
            mask_arr[rle_mask == 1] = fill
            mask = Image.fromarray(mask_arr)

        except ImportError:
            print("⚠ pycocotools not found. Install it to support RLE masks.")

    return mask


def coco_to_binary_masks(
    coco_json: str,
    images_dir: str,
    masks_dir: str,
    ignore_categories: list[str] | None = None,
) -> None:
    """
    Reads a COCO-format annotations file and writes one binary PNG mask
    per image into masks_dir.

    Supports:
      - Polygon segmentations
      - RLE segmentations

    Any annotation whose category name is in ignore_categories is erased
    from the final mask.
    """
    os.makedirs(masks_dir, exist_ok=True)
    ignore_categories = {c.lower() for c in (ignore_categories or ["inoculum"])}

    with open(coco_json) as f:
        coco = json.load(f)

    # category_id -> category_name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    # Group annotations by image_id
    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    skipped = 0
    for img_info in tqdm(coco["images"], desc="Generating masks"):
        img_id = img_info["id"]
        filename = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Start with black background
        mask = Image.new("L", (width, height), 0)

        annotations = ann_by_image.get(img_id, [])
        if not annotations:
            skipped += 1

        keep_anns = []
        ignore_anns = []

        for ann in annotations:
            cat_name = cat_id_to_name.get(ann.get("category_id"), "").lower()
            if cat_name in ignore_categories:
                ignore_anns.append(ann)
            else:
                keep_anns.append(ann)

        # First draw all kept categories as foreground
        for ann in keep_anns:
            mask = draw_annotation(mask, ann, width, height, fill=255)

        # Then erase ignored categories (e.g. inoculum)
        for ann in ignore_anns:
            mask = draw_annotation(mask, ann, width, height, fill=0)

        stem = Path(filename).stem
        out_path = os.path.join(masks_dir, f"{stem}.png")
        mask.save(out_path)

    print(f"\n✅ Saved {len(coco['images'])} masks to '{masks_dir}'")
    if skipped:
        print(f"⚠ {skipped} images had no annotations — blank masks were created.")



def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def process_export_folder(export_dir: Path) -> None:
    annotations_file = export_dir / "annotations" / "instances_default.json"
    images_dir = export_dir / "images" / "default"

    if not annotations_file.exists():
        print(f"⚠ Skipping {export_dir.name}: missing {annotations_file}")
        return

    if not images_dir.exists():
        print(f"⚠ Skipping {export_dir.name}: missing {images_dir}")
        return

    print(f"\n📁 Processing: {export_dir.name}")

    prefix = export_dir.name[:5]

    # Generate masks into a temporary folder first
    with tempfile.TemporaryDirectory() as tmp_mask_dir:
        coco_to_binary_masks(
            coco_json=str(annotations_file),
            images_dir=str(images_dir),
            masks_dir=tmp_mask_dir,
            ignore_categories=IGNORE_CATEGORIES,
        )

        # Collect stems of images that actually exist
        existing_image_stems = set()

        # Copy images into final output/images with short prefix
        for img_path in images_dir.iterdir():
            if img_path.is_file():
                existing_image_stems.add(img_path.stem)
                new_img_name = f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, OUTPUT_IMAGES_DIR / new_img_name)

        # Copy masks into final output/masks with short prefix
        # Only copy a mask if its corresponding image exists
        tmp_mask_dir = Path(tmp_mask_dir)
        for mask_path in tmp_mask_dir.glob("*.png"):
            if mask_path.stem not in existing_image_stems:
                print(f"⚠ Skipping mask without image: {mask_path.name}")
                continue

            new_mask_name = f"{prefix}_{mask_path.name}"
            shutil.copy2(mask_path, OUTPUT_MASKS_DIR / new_mask_name)
            
            
def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_ROOT}")

    recreate_dir(OUTPUT_IMAGES_DIR)
    recreate_dir(OUTPUT_MASKS_DIR)

    export_folders = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]

    if not export_folders:
        print(f"⚠ No export folders found in {INPUT_ROOT}")
        return

    for export_dir in sorted(export_folders):
        process_export_folder(export_dir)

    print("\n✅ Done.")
    print(f"Images saved to: {OUTPUT_IMAGES_DIR}")
    print(f"Masks saved to:  {OUTPUT_MASKS_DIR}")


if __name__ == "__main__":
    main()