from __future__ import annotations

import argparse
import warnings
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


Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


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
    if mask_np.size and int(mask_np.max()) <= 2:
        colors = {1: np.array([40, 220, 70]), 2: np.array([235, 70, 200])}
        for class_id, color in colors.items():
            pixels = mask_np == class_id
            overlay[pixels] = (0.5 * overlay[pixels] + 0.5 * color).astype(np.uint8)
    else:
        fg_pixels = mask_np > 127
        overlay[fg_pixels, 0] = np.clip(overlay[fg_pixels, 0].astype(int) * 0.5, 0, 255)
        overlay[fg_pixels, 1] = np.clip(overlay[fg_pixels, 1].astype(int) * 0.5 + 128, 0, 255)
        overlay[fg_pixels, 2] = np.clip(overlay[fg_pixels, 2].astype(int) * 0.5, 0, 255)
    return overlay.astype(np.uint8)


def save_rgb_image(path: Path, image_array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array.astype(np.uint8)).save(path)


def _repair_raw_16bit_tiff_tiles(image: Image.Image) -> None:
    if image.format != "TIFF" or image.mode != "I;16":
        return

    repaired_tiles = []
    for tile in image.tile:
        if tile[0] == "raw":
            tile = tile._replace(args=("I;16", 0, 1))
        repaired_tiles.append(tile)
    image.tile = repaired_tiles


def load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        _repair_raw_16bit_tiff_tiles(image)
        if image.format == "TIFF" and image.mode == "I;16":
            grayscale = np.asarray(image).astype(np.uint16, copy=False)
            grayscale_8bit = (grayscale >> 8).astype(np.uint8)
            return np.repeat(grayscale_8bit[..., np.newaxis], 3, axis=2)
        return np.array(image.convert("RGB"))


def predict_probabilities_on_image(
    model, image_path: Path, config: dict, device: torch.device
) -> np.ndarray:
    data_cfg = config["data"]
    transforms = get_val_transforms(
        data_cfg.get("image_size"),
        augmentations_config=config.get("augmentations", {}),
    )

    image_array = load_rgb_image(image_path)

    height, width = image_array.shape[:2]
    patching_cfg = config["patching"]
    patch_size = int(patching_cfg["patch_size"])
    stride = int(patching_cfg["stride"])
    xs = _compute_positions(width, patch_size, stride)
    ys = _compute_positions(height, patch_size, stride)

    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    num_classes = int(config.get("model", {}).get("num_classes", 3 if multiclass else 1))
    probability_sum = np.zeros(
        (num_classes, height, width) if multiclass else (height, width), dtype=np.float32
    )
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
            if multiclass:
                probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
            valid_height = min(patch_size, height - y)
            valid_width = min(patch_size, width - x)
            if multiclass:
                probability_sum[:, y : y + valid_height, x : x + valid_width] += probabilities[
                    :, :valid_height, :valid_width
                ]
            else:
                probability_sum[y : y + valid_height, x : x + valid_width] += probabilities[
                    :valid_height, :valid_width
                ]
            probability_count[y : y + valid_height, x : x + valid_width] += 1.0

    averaged_probabilities = probability_sum / np.clip(
        probability_count[None, ...] if multiclass else probability_count,
        a_min=1.0, a_max=None,
    )
    return averaged_probabilities


def probabilities_to_binary_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(np.uint8) * 255


def run_inference_on_image(
    model, image_path: Path, config: dict, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    multiclass = str(config.get("segmentation", {}).get("mode", "binary")).lower() == "multiclass"
    averaged_probabilities = predict_probabilities_on_image(model, image_path, config, device)
    if multiclass:
        output_mask = averaged_probabilities.argmax(axis=0).astype(np.uint8)
    else:
        threshold = float(config["inference"]["threshold"])
        output_mask = probabilities_to_binary_mask(averaged_probabilities, threshold)
    image_array = load_rgb_image(image_path)
    overlay = create_overlay(image_array, output_mask)
    return averaged_probabilities, output_mask, overlay


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
            if probabilities.ndim == 3:
                save_mask_image(Path(output_dir) / f"{image_path.stem}_prob_loci.png", probabilities[1] * 255.0)
                save_mask_image(Path(output_dir) / f"{image_path.stem}_prob_inoculum.png", probabilities[2] * 255.0)
            else:
                save_mask_image(Path(output_dir) / f"{image_path.stem}_prob.png", probabilities * 255.0)


if __name__ == "__main__":
    main()
