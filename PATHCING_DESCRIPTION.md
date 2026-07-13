# Patching Description

This document describes how patching works in the current training setup. Keep it updated whenever patch
generation, patch filtering, train/validation/test split handling, or patch visualization changes.

Note: the filename is intentionally `PATHCING_DESCRIPTION.md` to match the project request.

## Current Data Flow

The pipeline is binary segmentation for one target at a time. The active target is selected in `config.yaml`:

```yaml
segmentation:
  target: loci
```

The target controls which mask directory is used:

```yaml
paths:
  images_dir: data/images
  mask_dirs:
    loci: data/loci_masks
    inoculum: data/inoculum_masks
```

Images and masks are matched by filename stem. For example:

```text
data/images/IN183 YNA 48h x500 6.tif
data/loci_masks/IN183 YNA 48h x500 6.png
data/inoculum_masks/IN183 YNA 48h x500 6.png
```

The train/validation/test image membership comes from:

```yaml
split:
  mode: csv
  csv_path: data/image_splits.csv
```

The CSV has `filename,split` rows. The default split currently contains 20 train images, 4 validation images, and
5 held-out test images.

## Patch Geometry

The current default patch geometry is:

```yaml
patching:
  patch_size: 512
  stride: 256
  overlap: 256
```

Each model sample is a square `512 x 512` patch. With stride `256`, neighboring deterministic patches overlap by
50%.

For deterministic validation, test, inference, and stitched full-image evaluation, patch origins are computed with
`_compute_positions(length, patch_size, stride)`:

- if the image side is smaller than or equal to `patch_size`, the only origin is `0`;
- otherwise origins start at `0` and advance by `stride`;
- the final origin `length - patch_size` is always included, even when the stride does not land exactly there.

This guarantees full edge coverage. No right or bottom border is silently skipped.

## Training Patch Randomization

Training uses the newer epoch-specific patching method:

```yaml
patching:
  train:
    random_offset:
      enabled: true
      max_fraction_of_patch: 0.5
    scaled_context:
      enabled: true
      probability: 0.3
      max_scale: 2.0
      beta_alpha: 1.0
      beta_beta: 2.0
```

For each epoch, training patch records are regenerated with a deterministic random generator seeded by:

```text
train.seed + epoch
```

This means training patches change across epochs, but the exact sequence is reproducible.

When `random_offset.enabled` is true, the grid can shift in `x` and `y`. The largest offset is:

```text
min(stride - 1, round(patch_size * max_fraction_of_patch))
```

With the current config, this is `min(255, 256) = 255`, so each epoch may shift the training grid by 0 to 255 pixels
in each direction. The shifted grid still includes origin `0` and the final edge-covering position, so borders remain
represented.

## Scaled-Context Training Patches

For each training grid patch, scaled context may be sampled. With the current config, each kept candidate has a 30%
chance of becoming a scaled-context patch.

Normal patch:

```text
crop 512 x 512 from the source image
use it directly as the model input
```

Scaled-context patch:

```text
crop a larger source window around the same patch center
resize it back to 512 x 512
```

The scale is sampled from a beta distribution and capped by `max_scale: 2.0` and by available image boundaries.
For example, a scale near `2.0` means the source crop is close to `1024 x 1024`, then resized back to `512 x 512`.

Scaled context gives the model a wider field of view while keeping the tensor shape unchanged.

Image crops use the configured image resampling:

```yaml
image_resampling: lanczos
```

Mask crops use:

```yaml
mask_resampling: foreground_preserving
```

For masks, scaled crops are binarized before resizing, resized with a foreground-preserving mode, and thresholded
back to binary labels. This is meant to avoid erasing thin fungal structures during downsampling.

## Foreground Filtering

Candidate patch records are filtered by mask foreground:

```yaml
filter_empty_patches: true
mask_threshold: 127
min_foreground_pixels: 128
```

For every candidate patch, the matching target mask patch is binarized with `mask > 127`. If fewer than 128 pixels
are foreground, that patch record is discarded.

This filtering is applied when building train, validation, and test patch datasets. Therefore:

- patch-level train/validation/test metrics are measured on foreground-containing kept patches;
- stitched full-image validation/test metrics still cover the whole original image, because full-image evaluation
  tiles the image directly and averages predictions over all deterministic patch positions.

The stitched full-image metrics are the most representative image-level held-out scores.

## Train, Validation, And Test Behavior

Training split:

- uses only CSV `train` images;
- regenerates patch records every epoch;
- uses random grid offsets;
- may use scaled-context crops;
- applies foreground filtering.

Validation split:

- uses only CSV `validation` or `val` images;
- uses deterministic unshifted patch geometry;
- does not use scaled-context patches;
- applies foreground filtering for patch-level metrics;
- optionally runs stitched full-image validation at `train.per_image_validation_interval`;
- selects `best.pt` through the configured validation monitor.

Test split:

- uses only CSV `test` images;
- is evaluated after training by reloading `fold_0/best.pt`;
- does not influence training, scheduler decisions, or checkpoint selection;
- uses deterministic patch geometry and overlap averaging, matching `src.inference`;
- writes full-size binary masks to `test-evaluation/masks/` and overlays to `test-evaluation/overlays/`;
- writes one CSV row per image plus a final `mean` row in `test-evaluation/test_metrics.csv`, containing Dice/F1, IoU, precision, recall, clDice, and predicted foreground fraction; `threshold_metrics.csv` contains the same per-image threshold metrics;
- writes one threshold plot per metric, with one line per test image from thresholds 0.50 through 1.00.

## Patching Inspection And Visualization Script

Use the maintained patching inspection script:

```bash
venv/bin/python -m src.patching.explain --config config.yaml --target loci --epoch 1
```

This prints:

- matched image count;
- epoch and RNG seed;
- patch geometry;
- base candidate patch count before foreground filtering;
- kept and discarded patch counts;
- normal vs scaled-context patch counts;
- scale statistics;
- per-image source-crop-resolution table with raw `total` and `percent` summary rows;
- when `split.mode: csv`, additional tables with the same structure for train, validation, and test splits.

To inspect inoculum patching without editing `config.yaml`:

```bash
venv/bin/python -m src.patching.explain --config config.yaml --target inoculum --epoch 1
```

To generate a patch overlay visualization for one image:

```bash
venv/bin/python -m src.patching.explain \
  --config config.yaml \
  --target loci \
  --epoch 1 \
  --image "data/images/IN183 YNA 48h x500 6.tif"
```

And for inoculum:

```bash
venv/bin/python -m src.patching.explain \
  --config config.yaml \
  --target inoculum \
  --epoch 1 \
  --image "data/images/IN183 YNA 48h x500 6.tif"
```

The output files are written under:

```text
outputs/<project-name>/patching_explain/
```

The overlay colors are:

- green: normal patch footprint;
- orange: scaled-context patch footprint after resize to model patch size;
- red: larger source crop used for scaled-context patches.

The output filename includes the active target, for example:

```text
IN183 YNA 48h x500 6_loci_epoch_001_patches.png
IN183 YNA 48h x500 6_inoculum_epoch_001_patches.png
```

This allows loci and inoculum overlays for the same image to coexist.

## Other Patch Diagnostics

Foreground-count histogram:

```bash
venv/bin/python -m src.analyze_patches --config config.yaml
```

This also uses the active `segmentation.target`, so change `config.yaml` or use a copied config with
`target: inoculum` to inspect inoculum foreground distributions.

There is also `src/visualize_patch_grid.py`, but it is a simple manual helper with hard-coded paths and a naive
non-overlap grid. For the current training patching behavior, prefer `src.patching.explain`.

## Maintenance Checklist

If patching behavior changes, update this document and the README reference. In particular, update this document
when any of these change:

- `patch_size`, `stride`, or edge coverage logic;
- random-offset logic;
- scaled-context sampling;
- mask resampling behavior;
- foreground filtering;
- train/validation/test split behavior;
- qualitative or patch visualization commands.
