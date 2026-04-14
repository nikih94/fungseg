Build a clean, modular PyTorch project for binary semantic segmentation of fungi in RGB microscopy/macroscopy images.

Goal
- Train binary segmentation models on RGB images with associated binary masks.
- Images are stored in `data/images`
- Masks are stored in `data/masks`
- Image formats may include `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- Masks correspond to images by filename stem
- The first model to support is Unet++ with a ResNet50 encoder
- Later I want to swap in my own models without rewriting the rest of the pipeline

High-level requirements
1. Use PyTorch Dataset and DataLoader
2. Use k-fold cross-validation, where k is configurable
3. Split by ORIGINAL IMAGE, not by patch
   - all patches extracted from the same original image must stay in the same fold
   - do not mix patches from the same image between train and validation
4. Do NOT require CSV manifests for dataset indexing
   - discover files automatically by scanning `data/images` and `data/masks`
   - build records in memory
5. Keep the design modular:
   - model factory
   - loss factory
   - optimizer factory
   - scheduler factory
   - metrics module
   - patching module
   - trainer module
   - inference module
6. Put all important configuration in `config.yaml`

Core design choice
- Do not pre-split the dataset with train/val folders
- Do not split by patch
- Build a list of original image/mask pairs by scanning folders
- For each original image/mask pair, generate patch records in memory
- Each patch record must contain:
  - `source_id`: unique identifier of the ORIGINAL image (e.g. image stem)
  - `image_path`
  - `mask_path`
  - `x`
  - `y`
  - `patch_size`
- Use `source_id` as the grouping key for k-fold cross-validation
- Use patch records as training samples
- This ensures k-fold is applied at the image level while training is done at the patch level

Patch extraction requirements
- Implement patching from full-size images and masks
- Use these exact patching parameters from config defaults:

  PATCH_SIZE = 512
  OVERLAP = 128
  STRIDE = PATCH_SIZE - OVERLAP  # 384

- Empty-patch filtering:
  FILTER_EMPTY_PATCHES = True
  MASK_THRESHOLD = 127
  MIN_FOREGROUND_PIXELS = 1

Patching behavior
- Build patch records from each original image/mask pair
- Extract patches lazily in the Dataset (crop on-the-fly), not necessarily pre-save patch files to disk
- For each candidate patch:
  - crop the mask patch
  - convert mask to binary foreground using `mask > MASK_THRESHOLD`
  - if `FILTER_EMPTY_PATCHES=True`, keep the patch only if the foreground pixel count is >= `MIN_FOREGROUND_PIXELS`
- Ensure edge coverage:
  - include the final patch positions so no image area is lost even when width/height is not divisible by the stride
  - patch coordinates near the borders should be adjusted so the final crop is exactly `PATCH_SIZE x PATCH_SIZE`
  - do not discard image remainder regions

Cross-validation requirements
- Use k-fold cross-validation where `k` comes from `config.yaml`
- Apply the split at the original image level using `source_id`
- All patches from one original image must stay in exactly one fold
- Do not lose any original image if the dataset size is not divisible by `k`
- Fold sizes may differ slightly, but every original image must appear in validation exactly once across the full CV run
- For each fold:
  - create train patch records from training source images
  - create validation patch records from validation source images
  - create train and validation Datasets/DataLoaders from those records

Albumentations
- Use Albumentations for augmentation
- Keep train and validation transforms separate
- Use exactly this style and preserve the same operations:

  import albumentations as A
  import numpy as np
  import torch
  from albumentations.pytorch import ToTensorV2

  def get_train_transforms(image_size: Optional[int] = None) -> A.Compose:
      ops = []
      if image_size is not None:
          ops.append(A.Resize(image_size, image_size))
      ops.extend([
          A.HorizontalFlip(p=0.8),
          A.VerticalFlip(p=0.6),
          A.RandomRotate90(p=0.8),
          A.Affine(
              translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
              scale=(0.9, 1.1),
              rotate=(-30, 30),
              p=0.6,
          ),
          A.RandomBrightnessContrast(p=0.3),
          A.GaussNoise(p=0.2),
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(),
      ])
      return A.Compose(ops)

  def get_val_transforms(image_size: Optional[int] = None) -> A.Compose:
      ops = []
      if image_size is not None:
          ops.append(A.Resize(image_size, image_size))
      ops.extend([
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(),
      ])
      return A.Compose(ops)

Dataset behavior
- Create one dataset class for segmentation patches
- The dataset should receive:
  - a list of patch records
  - transforms
- In `__getitem__`:
  - load the full image and full mask
  - crop using record `(x, y, patch_size)`
  - convert image to RGB
  - convert mask to single-channel
  - binarize mask to `{0,1}`
  - apply albumentations transforms jointly to image and mask
  - return:
    - `image`
    - `mask`
    - optional metadata such as `source_id`, `x`, `y`
- Keep dataset logic independent from models and training logic

Model factory
- Create a modular model factory
- First supported model:
  - `unetplusplus_resnet50`
  - implemented with `segmentation_models_pytorch.UnetPlusPlus`
  - use:
    - `encoder_name="resnet50"`
    - `encoder_weights="imagenet"`
    - `in_channels=3`
    - `classes=1`
    - `activation=None`
- Also support optional torchvision baselines later, e.g.:
  - `deeplabv3_resnet50`
  - `fcn_resnet50`
- Implement `src/models/wrappers.py` with:

  def extract_logits(output):
      if isinstance(output, dict):
          return output["out"]
      return output

- Use `extract_logits()` everywhere in training/validation/inference so the rest of the code is model-agnostic

Losses
- Keep losses modular with a loss factory
- For now implement `BCEDiceLoss`
- `BCEDiceLoss` should:
  - combine `torch.nn.BCEWithLogitsLoss`
  - with Dice loss
  - operate on raw logits
- Make the combination configurable through weights, e.g.:
  - `bce_weight`
  - `dice_weight`
- Put custom combined loss in a dedicated module
- The trainer should not need to know loss internals
- Later I should be able to add focal loss, Tversky loss, etc. without rewriting the training loop

Metrics
- Keep metrics separate from losses
- Implement modular metric functions/classes for:
  - Dice score
  - IoU score
- Metrics should:
  - apply sigmoid to logits
  - threshold predictions
  - compare against binary masks
- Validation should report at least:
  - loss
  - Dice
  - IoU
- Best checkpoint selection should be based on a configurable validation metric, default `val_dice`

Optimizer factory
- Create a modular optimizer factory
- First support:
  - Adam
  - AdamW
  - SGD
- Read optimizer name and parameters from `config.yaml`

Scheduler factory
- Create a modular scheduler factory
- First support:
  - None
  - ReduceLROnPlateau
  - StepLR
  - CosineAnnealingLR
- Scheduler config should be fully controlled from `config.yaml`
- If using `ReduceLROnPlateau`, step it with the monitored validation metric or validation loss depending on config

Training engine
- Implement a trainer module responsible for training one fold
- The outer CV loop should be separate from the trainer
- For each fold:
  - initialize a fresh model
  - initialize a fresh optimizer
  - initialize a fresh scheduler
  - initialize loss
  - train for `epochs`
  - validate every epoch
  - save checkpoints
- Save at least:
  - `last.pt`
  - `best.pt`
- Organize checkpoints like:

  checkpoints/
    experiment_name/
      fold_0/
        best.pt
        last.pt
      fold_1/
        best.pt
        last.pt
      ...

- Also save per-fold metrics and final aggregated CV results:
  - mean Dice
  - std Dice
  - mean IoU
  - std IoU

Training script
- Create `src/train.py`
- Responsibilities:
  1. load `config.yaml`
  2. set seed and device
  3. scan dataset folders and match images with masks by stem
  4. build original image records
  5. build fold assignments using grouped k-fold on original images
  6. for each fold:
     - build train patch records
     - build val patch records
     - build datasets/loaders
     - build model/loss/optimizer/scheduler
     - run trainer
     - save fold metrics
  7. aggregate fold results at the end
  8. save a summary JSON/YAML/CSV of all fold metrics

Inference script
- Create `src/inference.py`
- Requirements:
  - load one trained checkpoint
  - run inference on full-size images from a given input path
  - use the same patching parameters as training
  - patch the full image, run the model on each patch, and stitch predictions back into a full-size probability map
  - for overlapping regions, average probabilities before thresholding
  - apply sigmoid to logits before stitching/thresholding
  - save:
    - probability map (optional)
    - final binary mask
  - support batch inference over a folder of images
  - do not require masks for inference

Configuration
- Put all important parameters into `config.yaml`
- The config must include at least these sections:

  project:
    name: fungi_segmentation

  paths:
    images_dir: data/images
    masks_dir: data/masks
    checkpoints_dir: checkpoints
    outputs_dir: outputs

  data:
    image_extensions: [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    patch_size: 512
    overlap: 128
    stride: 384
    filter_empty_patches: true
    mask_threshold: 127
    min_foreground_pixels: 1
    num_workers: 4
    pin_memory: true
    batch_size: 8

  cv:
    n_splits: 5
    shuffle_groups: true
    random_state: 42

  model:
    name: unetplusplus_resnet50
    in_channels: 3
    num_classes: 1
    encoder_name: resnet50
    encoder_weights: imagenet

  loss:
    name: bce_dice
    bce_weight: 0.5
    dice_weight: 0.5

  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.0001

  scheduler:
    name: reduce_on_plateau
    mode: max
    factor: 0.5
    patience: 5

  train:
    epochs: 50
    mixed_precision: true
    grad_clip: null
    monitor: val_dice
    monitor_mode: max
    threshold: 0.5

  inference:
    threshold: 0.5
    save_probabilities: false

Project structure
- Create this structure:

  project/
    config.yaml
    requirements.txt
    README.md
    src/
      train.py
      inference.py
      patching.py
      data/
        dataset.py
        discovery.py
        folds.py
      models/
        factory.py
        wrappers.py
      losses/
        factory.py
        combined.py
      metrics/
        segmentation.py
      optim/
        factory.py
      schedulers/
        factory.py
      engine/
        trainer.py
      utils/
        config.py
        checkpoint.py
        seed.py
        logging.py
        io.py

Module responsibilities
- `patching.py`
  - build original image records
  - build patch records
  - ensure complete edge coverage
  - filter empty patches
- `data/discovery.py`
  - scan folders
  - match image/mask pairs by stem
  - validate missing masks/images
- `data/folds.py`
  - create grouped folds from original image records
- `data/dataset.py`
  - segmentation patch dataset
- `models/factory.py`
  - build model from config
- `models/wrappers.py`
  - normalize output format with `extract_logits`
- `losses/factory.py`
  - build requested loss
- `losses/combined.py`
  - custom BCEDiceLoss
- `metrics/segmentation.py`
  - Dice and IoU
- `optim/factory.py`
  - optimizer builder
- `schedulers/factory.py`
  - scheduler builder
- `engine/trainer.py`
  - one-fold training/validation loop
- `utils/checkpoint.py`
  - save/load checkpoints
- `utils/config.py`
  - YAML loading
- `utils/seed.py`
  - reproducibility
- `utils/logging.py`
  - console/file logging

Implementation notes
- Use binary segmentation convention:
  - model output shape `(N, 1, H, W)`
  - masks as float tensors in `{0,1}`
- During training:
  - pass raw logits to the loss
- During metrics/inference:
  - apply sigmoid first
  - then threshold
- Use `model.train()` and `model.eval()` correctly
- Use `torch.no_grad()` in validation/inference
- Support mixed precision if CUDA is available
- Keep code clean, type-annotated, and easy to extend
- Avoid hardcoding model-specific logic in the trainer
- Avoid hardcoding fold logic in the dataset
- Avoid coupling patch extraction to one specific architecture

Deliverables
- Full project scaffold with working code
- Clear README with:
  - installation
  - training command
  - inference command
  - config explanation
- `requirements.txt`
- A minimal example command such as:
  - `python -m src.train --config config.yaml`
  - `python -m src.inference --config config.yaml --checkpoint path/to/best.pt --input path/to/images`

Please implement the project in a clean, research-friendly style suitable for later experimentation with custom segmentation models.