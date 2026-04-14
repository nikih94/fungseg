from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.data.dataset import get_val_transforms
from src.models.factory import build_model
from src.models.wrappers import extract_logits
from src.patching import _compute_positions, crop_and_pad_array
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_mask_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patch-based inference for fungi segmentation.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output", required=True, help="Directory for predictions.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_input_images(input_path: Path, image_extensions: list[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    extensions = {ext.lower() for ext in image_extensions}
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def create_overlay(original: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    overlay = np.array(original, copy=True)
    fg_pixels = mask_np > 127
    overlay[fg_pixels, 0] = np.clip(overlay[fg_pixels, 0].astype(int) * 0.5, 0, 255)
    overlay[fg_pixels, 1] = np.clip(overlay[fg_pixels, 1].astype(int) * 0.5 + 128, 0, 255)
    overlay[fg_pixels, 2] = np.clip(overlay[fg_pixels, 2].astype(int) * 0.5, 0, 255)
    return overlay.astype(np.uint8)


def save_rgb_image(path: Path, image_array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array.astype(np.uint8)).save(path)


def run_inference_on_image(
    model, image_path: Path, config: dict, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    threshold = float(config["inference"]["threshold"])
    transforms = get_val_transforms(
        data_cfg.get("image_size"),
        augmentations_config=config.get("augmentations", {}),
    )

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_array = np.array(rgb_image)

    height, width = image_array.shape[:2]
    patch_size = int(data_cfg["patch_size"])
    stride = int(data_cfg["stride"])
    xs = _compute_positions(width, patch_size, stride)
    ys = _compute_positions(height, patch_size, stride)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.float32)
    patch_coordinates = [(x, y) for y in ys for x in xs]

    model.eval()
    with torch.no_grad():
        patch_iterator = tqdm(
            patch_coordinates,
            desc=f"Patches | {image_path.name}",
            leave=False,
        )
        for x, y in patch_iterator:
            patch = crop_and_pad_array(image_array, x, y, patch_size)
            transformed = transforms(image=patch, mask=np.zeros((patch.shape[0], patch.shape[1]), dtype=np.float32))
            image_tensor = transformed["image"].unsqueeze(0).to(device)
            logits = extract_logits(model(image_tensor))
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
            valid_height = min(patch_size, height - y)
            valid_width = min(patch_size, width - x)
            probability_sum[y : y + valid_height, x : x + valid_width] += probabilities[:valid_height, :valid_width]
            probability_count[y : y + valid_height, x : x + valid_width] += 1.0

    averaged_probabilities = probability_sum / np.clip(probability_count, a_min=1.0, a_max=None)
    binary_mask = (averaged_probabilities >= threshold).astype(np.uint8) * 255
    overlay = create_overlay(image_array, binary_mask)
    return averaged_probabilities, binary_mask, overlay


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config["train"].get("device", "auto")))

    output_dir = ensure_dir(args.output)

    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    images = list_input_images(Path(args.input), config["data"]["image_extensions"])
    if not images:
        raise RuntimeError("No input images found for inference.")

    image_iterator = tqdm(images, desc="Images")
    for image_path in image_iterator:
        image_iterator.set_postfix(image=image_path.name)
        probabilities, binary_mask, overlay = run_inference_on_image(model, image_path, config, device)
        save_mask_image(Path(output_dir) / f"{image_path.stem}_mask.png", binary_mask)
        save_rgb_image(Path(output_dir) / f"{image_path.stem}_overlay.png", overlay)
        if config["inference"].get("save_probabilities", False):
            save_mask_image(Path(output_dir) / f"{image_path.stem}_prob.png", probabilities * 255.0)


if __name__ == "__main__":
    main()
