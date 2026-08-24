from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.inference.core import (
    predict_probabilities_on_image,
    probabilities_to_binary_mask,
    resolve_device,
)
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import config_for_persistence, load_config
from src.utils.io import ensure_dir, save_mask_image, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively infer fungal masks into a sibling <input-name>_masks directory "
            "while preserving the input folder structure."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--input", required=True, help="Input directory to process recursively.")
    return parser.parse_args()


def is_existing_mask(path: Path) -> bool:
    """Return whether a path has the old generated-mask naming convention."""
    return path.suffix.lower() == ".png" and path.stem.lower().endswith("_mask")


def list_recursive_input_images(input_dir: Path, image_extensions: list[str]) -> list[Path]:
    extensions = {extension.lower() for extension in image_extensions}
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and not is_existing_mask(path)
        ),
        key=lambda path: path.relative_to(input_dir),
    )


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_masks")


def output_path(image_path: Path, input_dir: Path, output_dir: Path, suffix: str) -> Path:
    relative_path = image_path.relative_to(input_dir)
    return output_dir / relative_path.parent / f"{relative_path.stem}_{suffix}.png"


def validate_no_output_collisions(images: list[Path], input_dir: Path) -> None:
    output_keys: dict[Path, Path] = {}
    for image_path in images:
        relative_path = image_path.relative_to(input_dir)
        key = relative_path.parent / relative_path.stem
        previous = output_keys.get(key)
        if previous is not None:
            raise ValueError(
                "Multiple source images would overwrite the same output masks: "
                f"{previous} and {image_path}."
            )
        output_keys[key] = image_path


def masks_from_probabilities(probabilities: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    segmentation_cfg = config.get("segmentation", {})
    multiclass = str(segmentation_cfg.get("mode", "binary")).lower() == "multiclass"
    if multiclass:
        combined = probabilities.argmax(axis=0).astype(np.uint8)
        classes = segmentation_cfg.get("classes", {})
        loci_id = int(classes.get("loci", 1))
        inoculum_id = int(classes.get("inoculum", 2))
        return {
            "mask": combined,
            "loci": (combined == loci_id).astype(np.uint8) * 255,
            "inoculum": (combined == inoculum_id).astype(np.uint8) * 255,
        }

    threshold = float(config["inference"]["threshold"])
    combined = probabilities_to_binary_mask(probabilities, threshold)
    masks = {"mask": combined}
    target = str(segmentation_cfg.get("target", "")).strip().lower()
    if target in {"loci", "inoculum"}:
        masks[target] = combined
    return masks


def probability_maps_from_probabilities(
    probabilities: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    segmentation_cfg = config.get("segmentation", {})
    multiclass = str(segmentation_cfg.get("mode", "binary")).lower() == "multiclass"
    if not multiclass or not config.get("inference", {}).get("save_probabilities", False):
        return {}

    classes = segmentation_cfg.get("classes", {})
    loci_id = int(classes.get("loci", 1))
    inoculum_id = int(classes.get("inoculum", 2))
    return {
        "prob_loci": probabilities[loci_id] * 255.0,
        "prob_inoculum": probabilities[inoculum_id] * 255.0,
    }


def run_recursive_mask_inference(
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_dir: str | Path,
) -> tuple[int, Path]:
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    output_dir = default_output_dir(input_dir)
    config = load_config(config_path)
    images = list_recursive_input_images(input_dir, config["data"]["image_extensions"])
    if not images:
        raise RuntimeError(f"No input images found under {input_dir}.")
    validate_no_output_collisions(images, input_dir)
    ensure_dir(output_dir)

    device = resolve_device(str(config["train"].get("device", "auto")))
    model = build_model(config["model"]).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    save_yaml(output_dir / "config.yaml", config_for_persistence(config))

    image_iterator = tqdm(images, desc="Images")
    for image_path in image_iterator:
        image_iterator.set_postfix(image=str(image_path.relative_to(input_dir)))
        probabilities = predict_probabilities_on_image(model, image_path, config, device)
        for suffix, mask in masks_from_probabilities(probabilities, config).items():
            save_mask_image(output_path(image_path, input_dir, output_dir, suffix), mask)
        for suffix, probability_map in probability_maps_from_probabilities(
            probabilities,
            config,
        ).items():
            save_mask_image(
                output_path(image_path, input_dir, output_dir, suffix),
                probability_map,
            )

    return len(images), output_dir


def main() -> None:
    args = parse_args()
    count, output_dir = run_recursive_mask_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_dir=args.input,
    )
    print(f"Saved inference outputs for {count} images under {output_dir}.")


if __name__ == "__main__":
    main()
