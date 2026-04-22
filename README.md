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

Training does the following:

1. Loads `config.yaml`.
2. Scans `data/images` and `data/masks`.
3. Builds original-image records.
4. Splits by original image, not by patch.
5. Generates patch records in memory.
6. Builds datasets, dataloaders, model, loss, optimizer, and scheduler.
7. Trains and validates each fold or manual split.
8. Saves checkpoints, logs, and metrics under `runs/<project>_<timestamp>/`.

Each run folder contains the merged config, per-fold checkpoints, TensorBoard logs, and CSV/JSON metric files.

## Inference

```bash
python -m src.inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/images \
  --output outputs/inference
```

Inference uses the same patch size and stride as training, predicts on overlapping patches, averages overlapping probabilities, and saves binary masks, overlay previews, and optional probability maps.

## Models

The current default model is SMP `Unet++` with an ImageNet-pretrained `resnet34` encoder.

Supported model names currently include:

- `unetplusplus_resnet18`
- `unetplusplus_resnet34`
- `unetplusplus_resnet50`
- `deeplabv3_resnet50`
- `fcn_resnet50`

For SMP `Unet++` models, the decoder can be configured with:

- attention via `model.decoder_attention_type`
- normalization via `model.decoder_normalization`

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
- `patch_size`: square crop size used for patch-based training and inference.
- `overlap`: overlap between neighboring patches.
- `stride`: patch step size. If omitted, it is derived from `patch_size - overlap`.
- `filter_empty_patches`: whether to discard patches without enough foreground.
- `mask_threshold`: grayscale threshold used to binarize masks.
- `min_foreground_pixels`: minimum positive pixels required to keep a patch when filtering is enabled.
- `num_workers`: dataloader worker count.
- `persistent_workers`: whether to keep dataloader workers alive across epochs.
- `prefetch_factor`: batches prefetched per worker when multiprocessing is enabled.
- `pin_memory`: whether dataloaders pin host memory for faster device transfer.
- `batch_size`: patch batch size.
- `image_size`: optional resize applied after patch extraction and before normalization.

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
- `alpha`: Tversky false-positive weight.
- `beta`: Tversky false-negative weight.
- `tversky_weight`: weight of the Tversky term in the combined loss.
- `soft_cldice_weight`: weight of the soft-clDice term in the combined loss.
- `iterations`: number of soft skeletonization iterations for clDice.
- `smooth`: smoothing term for Tversky-style calculations.
- `cldice_smooth`: smoothing term for soft-clDice.

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
- `threshold`: sigmoid threshold used for binary metrics.
- `enable_per_image_validation`: whether to run stitched full-image validation.
- `per_image_validation_interval`: epoch interval for full-image validation.
- `seed`: random seed used for reproducibility.
- `device`: target device such as `auto`, `cuda`, or `cpu`.
- `use_tqdm`: whether to show tqdm progress bars.

### `inference`

- `threshold`: sigmoid threshold used to export final binary masks.
- `save_probabilities`: whether to save probability maps in addition to binary masks.

## Repository Guide

- [DESCRIPTION.md](/home/niki/Desktop/funguy/fungseg/DESCRIPTION.md) gives a structural overview of the repository, explains how the modules connect, and lists the main dependencies.
