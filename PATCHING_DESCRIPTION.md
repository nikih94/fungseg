# Patching Description

This document describes the patch records used by training, validation, full-image evaluation, and inference. It is the detailed companion to the shorter patching section in `README.md`.

## Inputs and records

Images and masks are matched by filename stem. Binary mode creates one mask path per source image. Multiclass mode creates named loci and inoculum mask paths and requires both masks to match the image dimensions.

Each source becomes an `OriginalImageRecord` containing its source ID, paths, width, and height. Each model sample becomes a `PatchRecord` containing the source ID, top-left coordinate, model patch size, scale, and the source crop metadata. Patches retain their source identity so splits remain image-level.

## Deterministic patch geometry

The current `config.yaml`, `multiclass-config.yaml`, and `multiclass-segformer-config.yaml` use:

```yaml
patching:
  patch_size: 512
  overlap: 256
  stride: 256
```

Each model input is `512 × 512`. For a source dimension larger than the patch size, positions start at `0`, advance by `stride`, and always include the final position `length - patch_size`. This guarantees coverage of the right and bottom edges even when the stride does not land exactly on them. Sources smaller than one patch use position `0` and are padded for the model input.

This deterministic grid is used for validation, full-image validation, test evaluation, and inference. Overlapping predictions are averaged before the final binary threshold or multiclass `argmax` decision.

## Training patch randomization

Training uses the `patching.train` phase settings. In the current default config:

```yaml
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

For epoch `n`, patch randomness uses a NumPy generator seeded with `train.seed + n`. The grid can shift independently in x and y, while origin `0` and the final edge-covering position remain present. The offset is capped at `stride - 1` and at the configured fraction of the patch size.

For every candidate grid patch, scaled context is sampled before foreground filtering. When selected, a larger source crop is centered on the patch and resized back to the model patch size. The scale is sampled from the configured beta distribution, capped by `max_scale`, and reduced near image boundaries when necessary. Normal patches use scale `1.0`.

Image context uses `patching.image_resampling` (currently `lanczos`). Mask context is binarized first and uses `patching.mask_resampling` (currently `foreground_preserving`, implemented with area-style resampling followed by a positive-pixel threshold) to reduce loss of thin structures.

## Foreground filtering

The current defaults are:

```yaml
filter_empty_patches: true
mask_threshold: 127
min_foreground_pixels: 128
```

A mask pixel is foreground when it is greater than `mask_threshold`. A candidate is kept when its foreground count reaches `min_foreground_pixels`. In multiclass mode, the count is computed from the union of loci and inoculum foreground masks.

Filtering is applied while building train, validation, and test patch records. Patch-level metrics therefore describe the kept patch records. Full-image validation and test evaluation build their own deterministic grid over the complete source image, so they include background-only regions and are the image-level reference metrics.

## Phase behavior

| Phase | Grid | Random offset | Scaled context | Foreground filtering |
| --- | --- | --- | --- | --- |
| train | epoch-specific | configured, enabled by default | configured, enabled by default | yes |
| validation | deterministic | disabled by default | disabled by default | yes |
| test patch diagnostics | deterministic | disabled | disabled | yes |
| full-image validation/test/inference | deterministic | not applicable | not applicable | no |

Training regenerates training records each epoch. Validation records are built once per fold. Test patch records are used for fold diagnostics and counts; the actual test evaluation uses stitched full-image prediction through `src.test_evaluation`.

## Patching inspection

`src.patching.explain` is a binary, target-specific diagnostic. It reports candidate, kept, discarded, normal, and scaled-context counts; scale statistics; and source/split resolution tables.

```bash
python -m src.patching.explain --config config.yaml --target loci --epoch 1
python -m src.patching.explain --config config.yaml --target inoculum --epoch 1
```

To draw an overlay for one binary image:

```bash
python -m src.patching.explain \
  --config config.yaml \
  --target loci \
  --image "data/images/example.tif" \
  --epoch 1
```

The overlay is written to `outputs/<project>/patching_explain/`. Normal patch footprints are green, scaled model footprints are orange, and the larger source crops used for scaled-context records are red. The filename includes the target and epoch.

The diagnostic currently discovers a single binary target at a time. Multiclass training still uses the shared patching core, but this inspection command does not produce a combined loci/inoculum report.

## Maintenance rules

Update this document when any of these implementation details change:

- patch size, stride, overlap, or edge coverage;
- random-offset or scaled-context sampling;
- image or mask resampling;
- foreground thresholding or filtering;
- split-specific patch construction; or
- the behavior or output of `src.patching.explain`.
