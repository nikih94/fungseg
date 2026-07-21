# Repository Description

This repository contains a config-driven PyTorch segmentation pipeline. The main workflow discovers source images and masks, creates image-grouped patch records, trains a selected model, evaluates stitched full images, and stores auditable run artifacts.

## Repository map

- `config.yaml`: current binary U-Net++/ResNet34 experiment.
- `multiclass-config.yaml`: loci/inoculum multiclass U-Net++/ResNet34 experiment.
- `multiclass-segformer-config.yaml`: equivalent loci/inoculum multiclass SegFormer MiT-B3 experiment.
- `config_segformer_mit_b3.yaml`: legacy binary SegFormer experiment.
- `config-small-run.yaml`: legacy single-mask compatibility configuration.
- `requirements.txt`: runtime dependencies.
- `data/`: local images, masks, split CSV, and external evaluation data.
- `runs/`: training outputs.
- `outputs/`: inference and diagnostic outputs.
- `src/`: application code.
- `tests/`: unit and integration tests.

Generated data, archives, checkpoints, and exported images are not part of the application architecture.

## Application modules

### Entrypoints

- `src/train.py`: discovers data, builds splits, prepares loaders, constructs training components, runs folds, evaluates CSV test images, and optionally runs qualitative evaluation.
- `src/inference.py`: predicts on one image or a non-recursive directory using stitched overlapping patches.
- `src/test_evaluation.py`: evaluates a checkpoint on the CSV `test` split and writes full-image metrics and artifacts.
- `src/qualitative_evaluation.py`: compares manifest checkpoints on selected labeled crops.
- `src/in_folder_inference.py`: recursive binary inference that writes masks next to source images.
- `src/other_test_data_evaluation.py`: recursive inference for external paper image collections, preserving their folder structure in a results directory.

### `src/data/`

- `discovery.py`: matches images and masks by filename stem. Multiclass discovery also checks that every required mask exists and has the image dimensions.
- `folds.py`: implements CSV, manual train/validation, and grouped k-fold source splits.
- `dataset.py`: lazy patch loading, binary/multiclass mask composition, resizing, normalization, and Albumentations transforms.
- `sampling.py`: patch-distribution summaries saved for each training fold.

The dataset layer keeps source-image grouping separate from patch-level samples, so train/validation boundaries cannot be created by splitting patches from the same original image.

### `src/patching/`

- `core.py`: `OriginalImageRecord` and `PatchRecord`, deterministic edge-covering grids, epoch-specific training offsets, scaled-context crops, mask resampling, and foreground filtering.
- `explain.py`: binary target patch diagnostics and optional overlay generation.

The full patching contract is maintained in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

### `src/models/`

- `factory.py`: builds SMP U-Net++, SMP SegFormer, or torchvision FCN/DeepLabV3 models from config.
- `wrappers.py`: normalizes SMP and torchvision outputs through `extract_logits()`.
- `norms.py`: channel-wise 2D LayerNorm used for U-Net++ decoder normalization.

### `src/losses/`

- `factory.py`: maps configured loss names to implementations.
- `combined.py`: binary BCE/Dice, Tversky, clDice, soft-clDice, combined losses, and the multiclass CE/Dice/loci-clDice loss.

### `src/metrics/`

- `segmentation.py`: binary and multiclass Dice, IoU, precision, recall, and clDice calculations.
- `loss_components.py`: diagnostics for loss terms and component scores written with training metrics.

### `src/engine/`

- `trainer.py`: fold training and validation loop, optional stitched full-image validation, metric aggregation, scheduler stepping, checkpointing, TensorBoard logging, and checkpoint manifests.

### `src/optim/` and `src/schedulers/`

- `factory.py` in each directory builds the configured optimizer or learning-rate scheduler.

### `src/utils/`

- `config.py`: deep-merges YAML over defaults, derives patch stride, resolves the active mask directory, and preserves compatibility with older `paths.masks_dir` configs.
- `checkpoint.py`: checkpoint save/load helpers.
- `io.py`: JSON, YAML, CSV, mask, and directory helpers.
- `logging.py`: console/file logging setup.
- `seed.py`: reproducibility setup.

## Data and model flow

### Binary training

1. `train.py` resolves `segmentation.target` to one mask directory.
2. `discovery.py` matches image/mask stems.
3. `patching/core.py` creates source records and phase-specific patch records.
4. `folds.py` assigns source images to CSV, manual, or grouped k-fold splits.
5. `dataset.py` loads crops lazily and applies transforms.
6. The model, loss, optimizer, and scheduler factories build the training stack.
7. `trainer.py` runs patch-level training and validation, plus optional stitched full-image validation.
8. `test_evaluation.py` reloads each selected `best.pt` for CSV test evaluation.
9. `qualitative_evaluation.py` compares checkpoints when enabled.

### Multiclass training

Multiclass discovery requires a complete, dimension-matched loci and inoculum mask for every image. The dataset composes class-index masks in memory with inoculum precedence. Models output three logits; training uses softmax/argmax semantics, foreground-macro metrics, and loci-only clDice. Binary sigmoid thresholds and binary threshold sweeps are not used for this mode.

The U-Net++ and SegFormer multiclass configs intentionally share all non-model experiment settings. The model factory supplies either three-channel U-Net++/ResNet34 or three-channel SegFormer MiT-B3 logits to the same architecture-independent dataset, loss, trainer, test-evaluation, inference, and qualitative-evaluation paths.

### Inference

`inference.py` uses the same validation normalization and deterministic patch geometry as full-image validation. It averages overlapping sigmoid probabilities for binary models or softmax probabilities for multiclass models, then writes masks and overlays. Edge patches are padded for model input and cropped back to the valid image area before stitching.

## Splits and patch ownership

Splits are always made from original image IDs, never from patch records. CSV mode requires `filename,split` and non-empty train, validation, and test assignments. K-fold mode distributes source images across grouped validation folds. Manual mode creates one train/validation split from `val_source_ids`.

Training patch records are regenerated with `train.seed + epoch`; validation and full-image inference use deterministic geometry. Foreground filtering affects patch datasets and patch diagnostics, while stitched full-image evaluation covers all deterministic grid positions, including background-only regions.

## Run artifacts

Each run stores the merged config and split manifest at its root. The root also receives aggregate summaries and epoch/fold CSV files. Each fold stores checkpoint files, per-epoch history, patch-distribution diagnostics, TensorBoard data, and a checkpoint manifest recording checkpoint reason, epoch, monitor, and monitor value.

CSV test evaluation writes `test-evaluation/` with masks, overlays, `test_metrics.csv`, `threshold_metrics.csv`, and `summary.json`. Binary evaluation also writes per-metric threshold plots. Multiclass evaluation writes class metrics and `multiclass_metrics.png` without a threshold sweep.

Qualitative evaluation writes grids, selected crop metadata, crop-level metrics, and optional masks/probabilities under `qualitative_evaluation/`. K-fold runs additionally write the best-per-fold comparison artifacts.

## Configuration boundaries

Configuration is the source of truth for paths, segmentation mode, target/classes, patch geometry, augmentations, split strategy, model, loss, optimizer, scheduler, training, inference, test evaluation, and qualitative evaluation. `config.py` supplies defaults for omitted keys and supports old single-mask configs through `paths.masks_dir`; new configs should use `paths.mask_dirs` and an explicit segmentation target or mode.

The patch-specific options and their phase behavior belong in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md), rather than being duplicated here.

## Dependencies and tests

The runtime stack is PyTorch/torchvision, segmentation-models-pytorch, Albumentations, Pillow, NumPy, PyYAML, TensorBoard, tqdm, and Matplotlib. Tests cover configuration compatibility, patching and multiscale crops, model construction, inference, evaluation, qualitative evaluation, multiclass behavior, and dataloader cleanup.

Run the test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
