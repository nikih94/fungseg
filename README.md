# Fungi Segmentation

Modular PyTorch project for binary semantic segmentation of fungal networks in RGB microscopy or macroscopy images.

The training pipeline discovers image and mask pairs automatically, creates patch records in memory, trains on patches, and keeps all patches from the same original image in the same validation split.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Put source images in `data/images` and binary masks in `data/masks`.

- Images and masks are matched by filename stem.
- Supported image extensions come from `config.yaml`.
- No CSV manifest is required.

## Training

```bash
python -m src.train --config config.yaml
```

To train the SegFormer MiT-B3 variant with 512 pixel patches:

```bash
python -m src.train --config config_segformer_mit_b3.yaml
```

Training does the following:

1. Loads `config.yaml`.
2. Scans `data/images` and `data/masks`.
3. Builds original-image records.
4. Splits by original image, not by patch.
5. Generates validation patch records and epoch-specific training patch records in memory.
6. Builds datasets, dataloaders, model, loss, optimizer, and scheduler.
7. Trains and validates each fold or manual split.
8. Saves checkpoints, logs, and metrics under `runs/<project>_<timestamp>/`.
9. Runs qualitative checkpoint comparison when `qualitative_evaluation.enabled` is true.

Each run folder contains the merged config, `split_manifest.csv` / `split_manifest.json`, per-fold checkpoints,
TensorBoard logs, and CSV/JSON metric files. The split manifest records every train/validation source image per
fold, so k-fold and manual train/validation runs can be audited later.

Each fold always saves `best.pt`, `last.pt`, and `checkpoint_manifest.csv` with the metrics for each saved
checkpoint. When `train.best_interval_checkpoint.enabled` is true, it also saves configurable best-in-interval
snapshots such as `best_epochs_001_010.pt`.

Metric files include the optimized loss, hard-thresholded Dice/IoU, and loss-component diagnostics such as
soft Dice score, soft-clDice score, BCE, Tversky index, and weighted component contributions when those terms
are part of the configured loss.

Run-level metric outputs include:

- `cv_summary.json`: aggregate fold summary and final/best metrics.
- `fold_metrics.csv`: one row per fold with the selected best-epoch metrics.
- `epoch_metrics.csv`: one row per fold and epoch.
- `fold_*/metrics.csv` and `fold_*/metrics.json`: full per-fold epoch history.
- `fold_*/checkpoint_manifest.csv` and `.json`: saved checkpoints and their associated metrics.

## Inference

```bash
python -m src.inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_20260415_225352/fold_0/best.pt \
  --input data/images \
  --output outputs/inference
```

Inference uses the same patch size and stride as training, predicts on overlapping patches, averages overlapping probabilities, and saves binary masks, overlay previews, and optional probability maps.

### Recursive In-Folder Inference

```bash
python -m src.in_folder_inference \
  --config config_segformer_mit_b3.yaml \
  --checkpoint runs/fungi_segmentation_segformer_mit_b3_20260528_171100/fold_0/best.pt \
  --input data/images
```

This variant processes every supported image in the input folder and its subfolders, writes only binary masks next to the original images, and preserves each image stem with a `_mask.png` suffix. For example, `data/images/sample.tif` becomes `data/images/sample_mask.png`. Existing or generated `*_mask.png` files are skipped as inputs so reruns do not create nested mask names.

## Qualitative Evaluation

```bash
python -m src.qualitative_evaluation \
  --run-dir runs/fungi_segmentation_20260526_181338
```

Qualitative evaluation compares all checkpoints listed in each fold's `checkpoint_manifest.csv` on paired images and masks from `data/qualitative_evaluation/images` and `data/qualitative_evaluation/masks`.

For each qualitative image, it selects one configurable mostly-background crop with some foreground, predicts with all manifest checkpoints, and writes comparison grids and crop-level metrics under:

- `runs/<project>_<timestamp>/qualitative_evaluation/grids/`
- `runs/<project>_<timestamp>/qualitative_evaluation/eval_metrics.csv`
- `runs/<project>_<timestamp>/qualitative_evaluation/selected_crops.csv`

For k-fold cross-validation runs, qualitative evaluation also writes a compact best-per-fold comparison using one
`global_best` checkpoint per fold:

- `runs/<project>_<timestamp>/qualitative_evaluation/fold_comparison_grids/`
- `runs/<project>_<timestamp>/qualitative_evaluation/fold_comparison_metrics.csv`

Manual `train_val` runs skip these cross-fold files, so the additional CV comparison does not change non-CV runs.

Training runs this automatically at the end when `qualitative_evaluation.enabled` is true.

## Models

The current default model is SMP `Unet++` with an ImageNet-pretrained `resnet34` encoder.

Supported model names currently include:

- `unetplusplus_resnet18`
- `unetplusplus_resnet34`
- `unetplusplus_resnet50`
- `segformer_mit_b3`
- `deeplabv3_resnet50`
- `fcn_resnet50`

`segformer_mit_b3` uses SMP `Segformer` with an ImageNet-pretrained MiT-B3 encoder. The dedicated
`config_segformer_mit_b3.yaml` keeps the existing pipeline features but changes training patches to
512 pixels with 50% overlap.

For SMP `Unet++` models, the decoder can be configured with:

- attention via `model.decoder_attention_type`
- normalization via `model.decoder_normalization`

The default `config.yaml` currently uses `decoder_normalization: batchnorm`. The small-run config keeps
`instancenorm`. SegFormer does not use this Unet++ decoder-normalization option; its normalization comes from the
SMP SegFormer/MiT implementation.

Supported decoder normalization values:

- `batchnorm`
- `instancenorm`
- `layernorm`
- `identity`

## Config Overview

All important settings live in `config.yaml`.

### `project`

- `name`: experiment name used in output paths and run naming.

### `paths`

- `images_dir`: directory containing input images.
- `masks_dir`: directory containing binary segmentation masks.
- `runs_dir`: root directory for training runs, checkpoints, logs, and metrics.
- `outputs_dir`: root directory for inference outputs and exported artifacts.

### `data`

- `image_extensions`: file extensions accepted during dataset discovery.
- `num_workers`: dataloader worker count.
- `persistent_workers`: whether to keep dataloader workers alive across epochs.
- `prefetch_factor`: batches prefetched per worker when multiprocessing is enabled.
- `pin_memory`: whether dataloaders pin host memory for faster device transfer.
- `batch_size`: patch batch size.
- `image_size`: optional resize applied after patch extraction and before normalization.

### `patching`

- `patch_size`: square model crop size used for patch-based training, validation, and inference.
- `overlap`: overlap between neighboring deterministic patches.
- `stride`: patch step size. If omitted, it is derived from `patch_size - overlap`.
- `filter_empty_patches`: whether to discard training/validation patches without enough foreground.
- `mask_threshold`: grayscale threshold used to binarize masks.
- `min_foreground_pixels`: minimum positive pixels required to keep a patch when filtering is enabled.
- `image_resampling`: image resize filter used when scaled-context patches are resized back to `patch_size`.
- `mask_resampling`: mask resize mode. `foreground_preserving` is intended to preserve thin filament labels.
- `train.random_offset.enabled`: whether each epoch shifts the training grid by a deterministic random offset.
- `train.random_offset.max_fraction_of_patch`: maximum offset as a fraction of patch size, capped by stride.
- `train.scaled_context.enabled`: whether some training patches crop larger context and resize back to `patch_size`.
- `train.scaled_context.probability`: probability that a grid patch becomes scaled-context, usually `0.25`.
- `train.scaled_context.max_scale`: largest context crop multiplier, usually `2.0`.
- `train.scaled_context.beta_alpha` and `beta_beta`: Beta distribution parameters; defaults bias scales close to `1.0`.
- `validation.random_offset.enabled` and `validation.scaled_context.enabled`: normally false so validation/inference remain deterministic.

Training uses `train.seed + epoch` for patch randomness, so patch cuts and scaled-context choices change each epoch but remain reproducible.
Validation, inference, and stitched full-image validation use the deterministic unscaled grid.

To inspect patching for a config:

```bash
python -m src.patching.explain --config config.yaml
python -m src.patching.explain --config config.yaml --image "data/images/example.tif" --epoch 3
```

The explanation command prints patch counts, normal/scaled-context counts, scale statistics, and a per-source
source-crop-resolution table with a final percentage row across resolution bins. With `--image`, it saves an image
overlay under `outputs/<project>/patching_explain/`.

### `augmentations.normalize`

- `mean`: normalization mean used for image channels.
- `std`: normalization standard deviation used for image channels.

### `augmentations.train`

- `horizontal_flip_p`: probability of horizontal flips.
- `vertical_flip_p`: probability of vertical flips.
- `random_rotate_90_p`: probability of 90 degree rotations.
- `affine.translate_x`: horizontal translation range as relative fraction.
- `affine.translate_y`: vertical translation range as relative fraction.
- `affine.scale`: isotropic scale range.
- `affine.rotate`: rotation range in degrees.
- `affine.p`: probability of the affine transform.
- `random_brightness_contrast.brightness_limit`: brightness change range.
- `random_brightness_contrast.contrast_limit`: contrast change range.
- `random_brightness_contrast.p`: probability of brightness and contrast augmentation.
- `random_gamma.gamma_limit`: gamma range for illumination variation.
- `random_gamma.p`: probability of gamma augmentation.
- `clahe.clip_limit`: CLAHE contrast limit range.
- `clahe.tile_grid_size`: CLAHE grid size.
- `clahe.p`: probability of CLAHE augmentation.
- `blur.gaussian_blur_limit`: Gaussian blur kernel size range.
- `blur.gaussian_sigma_limit`: Gaussian blur sigma range.
- `blur.defocus_radius`: defocus radius range.
- `blur.defocus_alias_blur`: anti-alias blur range used by defocus.
- `blur.p`: probability of applying one blur variant.
- `gauss_noise_p`: probability of Gaussian noise augmentation.

### `cv`

- `n_splits`: number of folds for grouped cross-validation.
- `shuffle_groups`: whether to shuffle source images before fold assignment.
- `random_state`: random seed for grouped fold shuffling.

### `split`

- `mode`: split strategy. Supported values are `train_val` and `kfold`.
- `val_source_ids`: validation image identifiers used only in `train_val` mode.

Set `split.mode: kfold` to run grouped cross-validation. In this mode, `val_source_ids` is ignored and each
source image is used as validation in exactly one fold. The exact fold membership is written to the split
manifest files in the run directory.

### `model`

- `name`: model name passed to the model factory.
- `in_channels`: number of input image channels.
- `num_classes`: number of output channels. For binary segmentation this should stay `1`.
- `encoder_name`: encoder backbone name.
- `encoder_weights`: pretrained encoder weights, typically `imagenet` or `null`.
- `decoder_normalization`: decoder normalization for SMP `Unet++`.
- `decoder_channels`: decoder channel widths for SMP `Unet++`.
- `decoder_attention_type`: optional decoder attention block such as `scse`.

### `loss`

- `name`: loss name used by the loss factory.
- `bce_weight`: weight of the BCE term when `loss.name` is `bce_dice`.
- `dice_weight`: weight of the Dice term when `loss.name` is `bce_dice`.
- `soft_cldice_weight`: weight of the soft-clDice term when `loss.name` is `bce_dice_cldice` or `tversky_soft_cldice`.
- `iterations`: number of soft skeletonization iterations for clDice-based losses.
- `cldice_smooth`: smoothing term for soft-clDice and clDice-based combined losses.
- `alpha`: Tversky false-positive weight.
- `beta`: Tversky false-negative weight.
- `tversky_weight`: weight of the Tversky term in the combined loss.
- `smooth`: smoothing term for Tversky-style calculations.

### `optimizer`

- `name`: optimizer name such as `adam`, `adamw`, or `sgd`.
- `lr`: learning rate.
- `weight_decay`: weight decay coefficient.

### `scheduler`

- `name`: scheduler name such as `reduce_on_plateau`, `steplr`, `cosineannealinglr`, or `none`.
- `mode`: scheduler direction for monitored metrics, usually `max` or `min`.
- `factor`: multiplicative learning-rate reduction factor.
- `patience`: epochs to wait before reducing the learning rate.
- `min_lr`: lower bound for the learning rate.
- `monitor`: metric used when the scheduler depends on validation performance.

### `train`

- `epochs`: number of training epochs.
- `mixed_precision`: whether to use AMP when CUDA is available.
- `grad_clip`: optional gradient clipping value.
- `monitor`: metric used to select `best.pt`.
- `monitor_mode`: whether higher or lower values are considered better.
- `best_interval_checkpoint.enabled`: whether to save best-in-window checkpoint snapshots.
- `best_interval_checkpoint.interval_epochs`: epoch window size for interval-best snapshots.
- `threshold`: sigmoid threshold used for binary metrics.
- `enable_per_image_validation`: whether to run stitched full-image validation.
- `per_image_validation_interval`: epoch interval for full-image validation.
- `seed`: random seed used for reproducibility.
- `device`: target device such as `auto`, `cuda`, or `cpu`.
- `use_tqdm`: whether to show tqdm progress bars.

### `inference`

- `threshold`: sigmoid threshold used to export final binary masks.
- `save_probabilities`: whether to save probability maps in addition to binary masks.

### `qualitative_evaluation`

- `enabled`: whether training should run qualitative comparison after finishing.
- `data_root`: folder containing qualitative `images/` and `masks/` subfolders.
- `crop_patch_grid`: number of patch origins in the selected crop, such as `[3, 3]`.
- `min_foreground_ratio`: minimum foreground fraction for automatic crop selection.
- `max_foreground_ratio`: maximum foreground fraction for automatic crop selection.
- `selection_seed`: random seed for repeatable qualitative crop selection. Set to `null` to use the deterministic best-match crop.
- `max_checkpoints`: optional cap on manifest checkpoints to evaluate.

## Repository Guide

- [DESCRIPTION.md](/home/niki/fungseg/DESCRIPTION.md) gives a structural overview of the repository, explains how the modules connect, and lists the main dependencies.
