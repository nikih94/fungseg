from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from src.inference import (
    predict_probabilities_on_image,
    probabilities_to_binary_mask,
    resolve_device,
)
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.io import save_mask_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recursive in-folder binary mask inference for fungi segmentation."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--input", required=True, help="Input directory to process recursively.")
    return parser.parse_args()


def is_generated_mask(path: Path) -> bool:
    return path.suffix.lower() == ".png" and path.stem.lower().endswith("_mask")


def list_recursive_input_images(input_dir: Path, image_extensions: list[str]) -> list[Path]:
    extensions = {ext.lower() for ext in image_extensions}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not is_generated_mask(path)
    )


def mask_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.png")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    config = load_config(args.config)
    device = resolve_device(str(config["train"].get("device", "auto")))

    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    images = list_recursive_input_images(input_dir, config["data"]["image_extensions"])
    if not images:
        raise RuntimeError("No input images found for in-folder inference.")

    threshold = float(config["inference"]["threshold"])
    image_iterator = tqdm(images, desc="Images")
    for image_path in image_iterator:
        image_iterator.set_postfix(image=str(image_path.relative_to(input_dir)))
        probabilities = predict_probabilities_on_image(model, image_path, config, device)
        binary_mask = probabilities_to_binary_mask(probabilities, threshold)
        save_mask_image(mask_output_path(image_path), binary_mask)


if __name__ == "__main__":
    main()
