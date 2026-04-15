from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


# Simple test script: edit these values directly when needed.
IMAGE_PATH = Path("data/images/job_4_IN183 Agar 48h x500 5.tif")
OUTPUT_DIR = Path("outputs/patch_grid_debug")
PATCH_SIZE = 256

# Drawing style.
GRID_COLOR = (255, 0, 0)
GRID_WIDTH = 3
LABEL_PATCHES = True


def draw_patch_grid(
    image_path: Path,
    output_dir: Path,
    patch_size: int,
    grid_color: tuple[int, int, int] = GRID_COLOR,
    grid_width: int = GRID_WIDTH,
    label_patches: bool = LABEL_PATCHES,
) -> Path:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

    canvas = rgb_image.copy()
    draw = ImageDraw.Draw(canvas)

    xs = list(range(0, width, patch_size))
    ys = list(range(0, height, patch_size))

    patch_index = 0
    for y in ys:
        for x in xs:
            patch_index += 1
            x1 = min(x + patch_size - 1, width - 1)
            y1 = min(y + patch_size - 1, height - 1)
            draw.rectangle((x, y, x1, y1), outline=grid_color, width=grid_width)

            if label_patches:
                label = f"{patch_index}"
                text_x = min(x + 8, max(width - 20, 0))
                text_y = min(y + 8, max(height - 20, 0))
                draw.text((text_x, text_y), label, fill=grid_color)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_patch_grid.png"
    canvas.save(output_path)
    return output_path


def main() -> None:
    output_path = draw_patch_grid(
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        patch_size=PATCH_SIZE,
    )

    print(f"Image: {IMAGE_PATH}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Saved grid image to: {output_path}")


if __name__ == "__main__":
    main()
