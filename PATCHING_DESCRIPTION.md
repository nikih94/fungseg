# Patching Description

This document describes the patch records used by training, validation, full-image evaluation, and inference. It is the detailed companion to the shorter patching section in `README.md`.

## Inputs and records

Images and masks are matched by filename stem. Binary mode creates one mask path per source image. Multiclass mode creates named loci and inoculum mask paths and requires both masks to match the image dimensions.

Each source becomes an `OriginalImageRecord` containing its source ID, paths, width, and height. Each model sample becomes a `PatchRecord` containing the source ID, top-left coordinate, model patch size, scale, and the source crop metadata. Patches retain their source identity so splits remain image-level.

## Deterministic patch geometry

Each experiment defines `patching.patch_size`, `patching.overlap`, and `patching.stride` in its YAML. For a source dimension larger than the configured patch size, positions start at `0`, advance by `stride`, and always include the final position `length - patch_size`. This guarantees coverage of the right and bottom edges even when the stride does not land exactly on them. Sources smaller than one patch use position `0` and are padded to the configured model input size.

This deterministic grid is used for fast patch validation, full-image validation, test evaluation, and inference. Fast validation derives its stride from `patching.patch_size - validation.fast.overlap`; an overlap of zero therefore produces nominally non-overlapping patches. The final edge-covering position is still appended, so dimensions that are not divisible by patch size can create limited overlap at the right or bottom boundary.

Full-image validation, test evaluation, and inference use the main `patching.stride`. Their overlapping predictions are averaged before the final binary threshold or multiclass `argmax`. `validation.full_image.enabled` and `interval_epochs` control whether and how often the stitched validation pass runs. `selection: all` uses all validation sources; `selection: smallest_area` with positive `max_images` deterministically chooses the smallest areas and breaks ties by source ID.

## Training patch randomization

Training uses the `patching.train` phase settings. `random_offset` controls whether the grid shifts and caps that shift with `max_fraction_of_patch`. `scaled_context` independently controls enablement, sampling probability, maximum scale, beta-distribution parameters, and optional containment filtering. Exact values are experiment parameters and must be read from the active YAML rather than inferred from this guide.

For epoch `n`, patch randomness uses a NumPy generator seeded with `train.seed + n`. The grid can shift independently in x and y, while origin `0` and the final edge-covering position remain present. The offset is capped at `stride - 1` and at the configured fraction of the patch size.

For every candidate grid patch, scaled context is sampled before foreground filtering. When selected, a larger source crop is centered on the patch and resized back to the model patch size. The scale is sampled from the configured beta distribution, capped by `max_scale`, and reduced near image boundaries when necessary. Normal patches use scale `1.0`.

### Scaled-context containment filtering

After foreground filtering, `scaled_context.containment_filter` can remove redundant retained patches within each source image. Source-crop rectangles are processed from largest to smallest. A smaller crop is removed when `intersection_area / smaller_crop_area` is greater than or equal to `threshold` for a larger crop that remains in the plan. Equal-area crops do not suppress one another, records from different source images are never compared, and retained records preserve their original deterministic grid order.

`enabled` activates the filter, `threshold` must be in `(0, 1]`, and `preserve_normal_patches` protects scale-1 records from removal. Keeping `preserve_normal_patches: true` is recommended because a large source crop is downsampled to the model input size and therefore does not replace the fine detail in a normal patch. The compatibility default is disabled; checked-in experiment YAMLs explicitly choose their behavior. The filter applies only to the phase where it is configured and does not alter full-image validation, test stitching, or inference grids.

Image context uses the configured `patching.image_resampling`. Mask context is binarized first and uses `patching.mask_resampling`; the `foreground_preserving` option is implemented with area-style resampling followed by a positive-pixel threshold to reduce loss of thin structures.

## Foreground filtering

A mask pixel is foreground when it is greater than `mask_threshold`. A candidate is kept when its foreground count reaches `min_foreground_pixels`. In multiclass mode, the count is computed from the union of loci and inoculum foreground masks, plus join-mask foreground when `join_masks.merge_with_loci` is enabled. Missing optional join masks contribute no pixels.

Training and test diagnostic records use `patching.filter_empty_patches`. Fast validation overrides that choice with `validation.fast.foreground_only` but reuses `patching.mask_threshold` and `patching.min_foreground_pixels`. Its patch metrics therefore describe only the retained records. Full-image validation and test evaluation build their own unfiltered deterministic grid over each selected complete source, so they include background-only regions. `validation.full_image.monitor` combines full-image foreground Dice with hard clDice; in multiclass mode the clDice term is loci-specific.

## Phase behavior

| Phase | Grid | Random offset | Scaled context | Foreground filtering | Containment filtering |
| --- | --- | --- | --- | --- | --- |
| train | epoch-specific | as configured | as configured | as configured | as configured |
| fast validation | deterministic; `validation.fast.overlap` | no | no | `validation.fast.foreground_only` | no |
| test patch diagnostics | deterministic validation phase | as configured | as configured | as configured | as configured |
| full-image validation/test/inference | deterministic | not applicable | not applicable | no | no |

Training regenerates training records each epoch. Fast validation records are built once per fold. Test patch records are used for fold diagnostics and counts; the actual test evaluation uses stitched full-image prediction through `src.inference.test_evaluation`.

## Optional FIVES training patches

With `data.use_fives: true`, each matched FIVES image contributes exactly four non-overlapping patches arranged as a centered 2×2 square. For image width `W`, height `H`, and configured patch size `P`, the top-left origin is `((W - 2P) // 2, (H - 2P) // 2)` and the four coordinates are that origin plus `(0, 0)`, `(P, 0)`, `(0, P)`, and `(P, P)`. A 2048×2048 image with `P = 512` therefore uses `(512, 512)`, `(1024, 512)`, `(512, 1024)`, and `(1024, 1024)`.

These records always use scale `1.0`; they do not use the fungal training grid, random offsets, scaled context, or foreground filtering. The same records are reused every epoch and pass through the configured `augmentations.train` pipeline on access. They are concatenated only with training data and never appear in validation, test, inference, qualitative evaluation, or fungal split manifests. Multiclass vessel foreground is encoded as loci class `1` with no inoculum pixels.

## Patching inspection

`src.patching.explain` is a binary, target-specific diagnostic. It reports the active containment settings plus candidate, kept, total discarded, foreground-filtered, containment-filtered, normal, and scaled-context counts; scale statistics; and source/split resolution tables. The resolution-table bins span from the configured model patch size through the training `scaled_context.max_scale` source-crop size.

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

FIVES geometry has a separate visual check:

```bash
python -m src.visualize_fives_patches --config config.yaml --image 60_D.png
```

## Maintenance rules

Update this document when any of these implementation details change:

- the meaning or interaction of patch size, stride, overlap, or edge coverage;
- random-offset or scaled-context sampling;
- image or mask resampling;
- foreground thresholding or filtering;
- split-specific patch construction; or
- the behavior or output of `src.patching.explain`.

Changing only experiment parameter values in a YAML file does not require copying those values into this guide.
