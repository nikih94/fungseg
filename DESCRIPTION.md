# Repository Description

This file explains the core structure of the repository, why each main folder exists, how the training and inference pipeline is wired together, and which dependencies are central to the project.

## Top-Level Layout

- `config.yaml`: main experiment configuration used by training and inference.
- `README.md`: user-facing quickstart and concise config reference.
- `DESCRIPTION.md`: repository structure and architecture guide.
- `requirements.txt`: Python dependencies needed to run the project.
- `data/`: local dataset storage for raw images, masks, and any auxiliary data folders.
- `runs/`: training outputs created at runtime, including fold checkpoints, logs, and metric files.
- `outputs/`: exported inference outputs and generated artifacts.
- `src/`: all application code for data loading, model building, training, evaluation, and utilities.

## `src/` Structure

### `src/train.py`

Main training entrypoint.

Responsibilities:

- load configuration
- set seed and device
- discover image and mask pairs
- build grouped splits
- generate patch records
- create datasets and dataloaders
- build model, loss, optimizer, and scheduler
- launch one trainer per fold or split
- aggregate metrics across folds

### `src/inference.py`

Main inference entrypoint.

Responsibilities:

- load one trained checkpoint
- build the model from config
- tile full-size images into patches
- run prediction patch by patch
- average overlapping probabilities
- write binary masks, overlays, and optional probability maps

### `src/patching.py`

Core patch-generation logic.

Why it matters:

- turns original image and mask pairs into `OriginalImageRecord` objects
- creates patch coordinates with full edge coverage
- filters empty patches based on mask foreground content
- preserves the distinction between original-image grouping and patch-level training samples

### `src/data/`

Dataset and split logic live here.

- `dataset.py`: patch dataset plus Albumentations train and validation transforms.
- `discovery.py`: scans image and mask directories and matches files by stem.
- `folds.py`: creates grouped k-fold splits or a manual train/validation split.

This folder is important because it keeps dataset discovery, patch sampling, and split strategy separate from model code.

### `src/models/`

Model construction and output normalization live here.

- `factory.py`: builds segmentation models from config.
- `wrappers.py`: exposes `extract_logits()` so the rest of the pipeline can treat SMP and torchvision models uniformly.
- `norms.py`: custom 2D LayerNorm helper used when `decoder_normalization` is set to `layernorm`.

This folder is important because it keeps model-specific details out of the training loop.

### `src/losses/`

Loss functions are organized here.

- `factory.py`: selects the requested loss from config.
- `combined.py`: contains custom segmentation losses such as BCE+Dice, Tversky, clDice variants, and the combined Tversky + soft-clDice loss.

This folder makes it easy to swap training objectives without rewriting the trainer.

### `src/metrics/`

Validation metrics live here.

- `segmentation.py`: Dice and IoU metrics for patch-level and stitched-image evaluation.

This folder stays separate from losses so monitoring and optimization remain independent.

### `src/optim/`

- `factory.py`: builds optimizers from config.

This isolates optimizer choice and hyperparameters from training orchestration.

### `src/schedulers/`

- `factory.py`: builds schedulers from config.

This keeps learning-rate policy selection modular and configurable.

### `src/engine/`

- `trainer.py`: one-fold training and validation engine.

Why it matters:

- runs epoch loops
- computes patch-level metrics
- optionally runs full-image stitched validation
- tracks the monitored metric
- saves `best.pt` and `last.pt`
- writes per-fold metric history

### `src/utils/`

General support code used across the project.

- `config.py`: loads YAML config and merges it with defaults.
- `checkpoint.py`: saves and loads checkpoints.
- `logging.py`: configures console and file logging.
- `seed.py`: sets random seeds for reproducibility.
- `io.py`: small filesystem and serialization helpers.

### Helper Scripts

These are useful but not central to the main training loop:

- `analyze_patches.py`: patch inspection or debugging utility.
- `visualize_patch_grid.py`: visualization helper for understanding patch coverage.

## How Everything Is Wired Together

### Training Flow

1. `src/train.py` loads `config.yaml` through `src/utils/config.py`.
2. `src/data/discovery.py` matches images and masks by filename stem.
3. `src/patching.py` creates original-image records containing source identity and image size.
4. `src/data/folds.py` splits by original image, not by patch.
5. `src/patching.py` creates patch records for the selected train and validation source images.
6. `src/data/dataset.py` loads full images lazily, crops patches on the fly, applies Albumentations, and returns tensors.
7. `src/models/factory.py` builds the requested segmentation model.
8. `src/losses/factory.py`, `src/optim/factory.py`, and `src/schedulers/factory.py` build the training components from config.
9. `src/engine/trainer.py` runs training, validation, checkpointing, and metric export.
10. `src/train.py` aggregates fold metrics and writes run summaries.

### Inference Flow

1. `src/inference.py` loads config and a checkpoint.
2. `src/models/factory.py` rebuilds the model architecture.
3. The full image is tiled using the same patch geometry defined in config.
4. Each patch is normalized with validation transforms and passed through the model.
5. Overlapping patch probabilities are averaged back into a full-size probability map.
6. The final binary mask is thresholded and written to disk.

## Why the Folder Boundaries Matter

- The data code does not need to know which model is being trained.
- The trainer does not need to know loss internals.
- The model factory hides architecture-specific details.
- The patching code enforces the image-level grouping rule.
- The utils layer keeps repetitive infrastructure code out of the core logic.

That separation is what makes it practical to swap models, losses, schedulers, and split strategies without rewriting the whole pipeline.

## Main Dependencies

- `torch`: model definition, tensor operations, optimization, AMP, and checkpoint loading.
- `torchvision`: optional segmentation baselines such as FCN and DeepLabV3.
- `segmentation-models-pytorch`: main implementation source for Unet++ and encoder backbones.
- `albumentations`: image and mask augmentations for training and validation.
- `numpy`: array manipulation for masks, patches, stitching, and metric preparation.
- `Pillow`: image file loading and saving.
- `PyYAML`: config parsing.
- `tensorboard`: experiment logging.
- `tqdm`: progress bars for training and inference.
- `matplotlib`: visualization utilities.
- `scikit-learn`: installed dependency, though the current split implementation is custom and lightweight.

## Current Design Priorities

- binary segmentation with output shape `(N, 1, H, W)`
- patch-based training with full-image grouping
- modular factories for models, losses, optimizers, and schedulers
- reproducible experiment runs through config-driven behavior
- clean separation between training, inference, data preparation, and utilities
