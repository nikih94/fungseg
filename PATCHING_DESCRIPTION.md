# Patching Description

This document describes the patch records used by training, validation, full-image evaluation, and inference. It is the detailed companion to the shorter patching section in `README.md`.

## Inputs and records

Images and masks are matched by filename stem. Binary mode creates one mask path per source image. Multiclass mode creates named loci and inoculum mask paths and requires both masks to match the image dimensions.

Each source becomes an `OriginalImageRecord` containing its source ID, paths, width, and height. Uncached workflows use `PatchRecord`. Cached training adds immutable `StaticPatchRecord` entries for larger source regions and epoch-specific `CachedTrainingCropRecord` entries for final model crops. Every form retains source identity so splits remain image-level.

For cache-enabled Soft-clDice training with `iterations_csv: null`, each static region stores the exact crisp-target iteration requirement plus `loss.static_patch_iterations.margin_iterations`, rounded upward to `round_up_to`. Multiclass calculation uses effective class-1 loci after optional join merging and inoculum precedence. The static value conservatively covers every final crop selected from that region. A configured CSV explicitly overrides these values with the existing per-source mapping. `loss.iterations` remains the fallback for FIVES, uncached training, or disabled automatic calculation. Batches continue grouping equal counts.

## Deterministic patch geometry

Each experiment defines `patching.patch_size`, `patching.overlap`, and `patching.stride` in its YAML. For a source dimension larger than the configured patch size, positions start at `0`, advance by `stride`, and always include the final position `length - patch_size`. This guarantees coverage of the right and bottom edges even when the stride does not land exactly on them. Sources smaller than one patch use position `0` and are padded to the configured model input size.

This deterministic grid is used for full-image validation, test evaluation, and inference. Full-image validation and test evaluation always use `patching.patch_size // 2`, giving 50% nominal overlap for the even patch sizes supported by the experiments. Inference uses the configured `patching.stride`. The final edge-covering position is still appended, so dimensions that are not divisible by patch size can create limited additional overlap at the right or bottom boundary.

Validation and test evaluation use `patching.patch_size // 2` regardless of the main `patching.stride`; inference uses the main stride. Their overlapping predictions are averaged before the final binary threshold or multiclass `argmax`. `validation.start_epoch` delays full-image validation until the configured epoch. Training requires `validation.full_image.enabled: true` and `interval_epochs: 1`. During validation, every half-stride image patch is inferred exactly once and contributes to stitching. `val_loss` indexes those same logits only at strict Cartesian-grid starts `0, P, 2P, ...` for patch size `P`, stopping where a complete patch no longer fits. If a source dimension is at most `P`, start `0` is used with normal padding. Thus final edge-covering overlap patches preserve full-image coverage but do not duplicate loss pixels; residual strips smaller than `P` are represented in stitched metrics rather than patch loss. `selection: all` uses all validation sources; `selection: smallest_area` with positive `max_images` deterministically chooses the smallest areas and breaks ties by source ID. With `validation.full_image.patch_cache.enabled: true`, all RGB patches are cached, while targets and adaptive loss metadata are stored only for the strict loss subset.

## Training patch randomization

Cache-enabled training places static anchors on the configured `stride` grid and stores square regions of side `patch_size + overlap`. The extra extent is centered around interior anchors where boundaries allow it. Every epoch, each interior region independently samples x and y for a final `patch_size` crop. Motion is capped by `overlap`, `stride - 1`, `random_offset.max_fraction_of_patch`, and valid source bounds. Origin and final-edge anchors remain fixed on their edge axis. Randomness is reproducible from the training seed, fold, epoch, source ID, and static-region index.

When static caching is disabled, training retains the existing epoch-shifted grid and optional scaled-context behavior. Static caching and training scaled context are configuration-incompatible; checked-in scaled-context experiments explicitly disable the cache.

For every candidate grid patch, scaled context is sampled before foreground filtering. When selected, a larger source crop is centered on the patch and resized back to the model patch size. The scale is sampled from the configured beta distribution, capped by `max_scale`, and reduced near image boundaries when necessary. Normal patches use scale `1.0`.

### Scaled-context containment filtering

After foreground filtering, `scaled_context.containment_filter` can remove redundant retained patches within each source image. Source-crop rectangles are processed from largest to smallest. A smaller crop is removed when `intersection_area / smaller_crop_area` is greater than or equal to `threshold` for a larger crop that remains in the plan. Equal-area crops do not suppress one another, records from different source images are never compared, and retained records preserve their original deterministic grid order.

`enabled` activates the filter, `threshold` must be in `(0, 1]`, and `preserve_normal_patches` protects scale-1 records from removal. Keeping `preserve_normal_patches: true` is recommended because a large source crop is downsampled to the model input size and therefore does not replace the fine detail in a normal patch. The compatibility default is disabled; checked-in experiment YAMLs explicitly choose their behavior. The filter applies only to the phase where it is configured and does not alter full-image validation, test stitching, or inference grids.

Image context uses the configured `patching.image_resampling`. Mask context is binarized first and uses `patching.mask_resampling`; the `foreground_preserving` option is implemented with area-style resampling followed by a positive-pixel threshold to reduce loss of thin structures.

## Foreground filtering

A mask pixel is foreground when it is greater than `mask_threshold`. For cached training, the epoch crop is selected first and foreground is counted only inside that final model-sized crop; foreground elsewhere in the static region cannot retain it. A candidate is kept when its final-crop count reaches `min_foreground_pixels`. In multiclass mode, the count is computed from the union of loci and inoculum foreground masks, plus join-mask foreground when `join_masks.merge_with_loci` is enabled. Missing optional join masks contribute no pixels.

`patching.train.background_only` can add a deterministic sample of strictly background-only candidates after foreground and containment filtering. For each source image, `percentage_of_foreground` is applied to that source's retained valid-foreground count and rounded up to the next whole patch. When the feature is enabled and the source has any zero-foreground candidate, the quota is at least one even when the percentage calculation is zero or the source has no retained foreground; it remains capped by the available background candidates. Patches with a nonzero foreground count below `min_foreground_pixels` are still discarded rather than treated as background-only. Selection is reproducible from `train.seed`, the epoch, and the source ID, and records retain deterministic grid order.

This quota is training-only and has no effect when `filter_empty_patches: false`. It does not change full-image validation, test evaluation, or inference.

Training and test diagnostic records use `patching.filter_empty_patches`. Full-image validation and test evaluation build their own unfiltered deterministic grid, so they include background-only regions. Validation reports stitched Dice/IoU and Zhang hard-clDice. Each immutable target skeleton is cached for the fold, and each stitched prediction is skeletonized once per source and validation epoch for reuse across hard-clDice-based monitors; training batches do not compute hard-clDice. It also reports `val_loss` over every strict non-overlapping loss patch, including background-only patches; `min_foreground_pixels` is not consulted. Each loss call is multiplied by its selected-patch count before epoch aggregation, so an inference batch containing fewer loss positions cannot receive equal weight to a fuller one. With `validation.full_image.soft_cldice_foreground_only`, only target-foreground samples enter the soft-clDice term; CE and Dice retain every selected loss patch. The same forward-pass logits feed loss and stitching.

## Phase behavior

| Phase | Grid | Random offset | Scaled context | Foreground filtering | Containment filtering | Background-only quota |
| --- | --- | --- | --- | --- | --- | --- |
| cached train | static stride anchors; epoch-specific interior crops | independent per region; edges fixed | incompatible | final crop | no | as configured |
| uncached train | epoch-specific | as configured | as configured | as configured | as configured | as configured |
| test patch diagnostics | deterministic validation phase | as configured | as configured | as configured | as configured | no |
| cached full-image validation | half-stride RGB stitching grid; strict `patch_size` loss subset | not applicable | not applicable | no | no | no |
| test evaluation | deterministic; `patching.patch_size // 2` | not applicable | not applicable | no | no | no |
| inference | deterministic; `patching.stride` | not applicable | not applicable | no | no | no |

When `data.train_patch_cache.enabled` is true, training builds one atomic run-level uint8 memmap cache before the first fold. It contains the union of sources that appear in any fold training set, while each fold exposes only its own source IDs. RGB and applicable masks are stored separately so identical local coordinates are applied before multiclass composition. Epoch planning reads only cached masks for randomization, foreground filtering, and per-source background selection. The cache is reused by every epoch and fold, then removed after the run.

When `validation.full_image.patch_cache.enabled` is true, each fold builds a separate atomic cache only for its selected validation sources. It stores exact model-sized RGB and composed-target patches at half-patch stride, full targets and optional join masks for stitched metrics, and per-patch adaptive Soft-clDice counts when enabled. It includes every grid position, including background. The cache is removed in fold cleanup; incomplete-fold resume removes stale data and rebuilds it. The direct validation path remains available for legacy configs. Test patch records and test evaluation are unchanged and do not use this cache.

## Optional FIVES training patches

With `data.use_fives: true`, each matched FIVES image contributes exactly four non-overlapping patches arranged as a centered 2×2 square. For image width `W`, height `H`, and configured patch size `P`, the top-left origin is `((W - 2P) // 2, (H - 2P) // 2)` and the four coordinates are that origin plus `(0, 0)`, `(P, 0)`, `(0, P)`, and `(P, P)`. A 2048×2048 image with `P = 512` therefore uses `(512, 512)`, `(1024, 512)`, `(512, 1024)`, and `(1024, 1024)`.

These records always use scale `1.0`; they do not use the fungal training grid, random offsets, scaled context, or foreground filtering. The same records are reused every epoch and pass through the configured `augmentations.train` pipeline on access. They are concatenated only with training data and never appear in validation, test, inference, qualitative evaluation, or fungal split manifests. Multiclass vessel foreground is encoded as loci class `1` with no inoculum pixels.

## Patching inspection

`src.patching.explain` is a binary, target-specific diagnostic. With caching enabled it outlines static regions in gray and reports their count and size in addition to epoch-selected final crops. It reports active containment and background-only settings plus candidate, kept, total discarded, foreground-filtered, containment-filtered, normal, scaled-context, and background-only counts; scale statistics; and source/split resolution tables. Each source table includes a `background` column. The resolution-table bins span from the configured model patch size through the training `scaled_context.max_scale` source-crop size.

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

The overlay is written to `outputs/<project>/patching_explain/`. Static cache regions are gray, normal final crops are green, scaled model footprints are orange, selected background-only crops are blue, and larger scaled-context source crops are red. The filename includes the target and epoch.

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
