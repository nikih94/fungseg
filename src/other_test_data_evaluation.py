from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT_DIR = Path("data/other-test-data")
DEFAULT_RESULTS_DIR_NAME = "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fungi segmentation checkpoint on images from other papers."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--model",
        "--model-path",
        "--checkpoint",
        dest="model_path",
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Root directory containing the other-paper images (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Results directory (default: <input>/results).",
    )
    return parser.parse_args()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def list_input_images(input_dir: Path, output_dir: Path, image_extensions: list[str]) -> list[Path]:
    """List source images recursively while ignoring the generated results tree."""
    extensions = {extension.lower() for extension in image_extensions}
    resolved_output_dir = output_dir.resolve()
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and not _is_inside(path.resolve(), resolved_output_dir)
        ),
        key=lambda path: path.relative_to(input_dir),
    )


def result_path(image_path: Path, input_dir: Path, results_dir: Path, suffix: str) -> Path:
    relative_path = image_path.relative_to(input_dir)
    return results_dir / relative_path.parent / f"{relative_path.stem}_{suffix}.png"


def evaluate_other_test_data(
    config_path: str | Path,
    model_path: str | Path,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path | None = None,
) -> int:
    from tqdm.auto import tqdm

    from src.inference import resolve_device, run_inference_on_image, save_rgb_image
    from src.models.factory import build_model
    from src.utils.checkpoint import load_checkpoint
    from src.utils.config import load_config
    from src.utils.io import ensure_dir, save_mask_image

    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    results_dir = Path(output_dir).resolve() if output_dir is not None else input_dir / DEFAULT_RESULTS_DIR_NAME
    if results_dir == input_dir:
        raise ValueError("The results directory must be different from the input directory.")
    ensure_dir(results_dir)

    config = load_config(config_path)
    device = resolve_device(str(config["train"].get("device", "auto")))

    model = build_model(config["model"]).to(device)
    load_checkpoint(model_path, model, map_location=device)

    images = list_input_images(input_dir, results_dir, config["data"]["image_extensions"])
    if not images:
        raise RuntimeError(f"No input images found under {input_dir}.")

    image_iterator = tqdm(images, desc="Images")
    for image_path in image_iterator:
        relative_path = image_path.relative_to(input_dir)
        image_iterator.set_postfix(image=str(relative_path))
        _, binary_mask, overlay = run_inference_on_image(model, image_path, config, device)
        save_mask_image(result_path(image_path, input_dir, results_dir, "mask"), binary_mask)
        save_rgb_image(result_path(image_path, input_dir, results_dir, "overlay"), overlay)

    return len(images)


def main() -> None:
    args = parse_args()
    count = evaluate_other_test_data(
        config_path=args.config,
        model_path=args.model_path,
        input_dir=args.input,
        output_dir=args.output,
    )
    results_dir = Path(args.output) if args.output else Path(args.input) / DEFAULT_RESULTS_DIR_NAME
    print(f"Saved masks and overlays for {count} images under {results_dir}.")


if __name__ == "__main__":
    main()
