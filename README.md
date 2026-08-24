# Fungi Segmentation

Modular PyTorch pipeline for patch-based semantic segmentation of fungal networks in RGB microscopy or macroscopy images. It supports binary segmentation of one target at a time and multiclass segmentation of loci plus inoculum.

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.train --config config.yaml
```

The checked-in `config.yaml` is the primary binary U-Net++/ResNet34 experiment. Other maintained configurations are:

- `multiclass-config.yaml`: three classes—background, loci, and inoculum—with U-Net++/ResNet34.
- `multiclass-segformer-mit-b1-refinement-config.yaml`: multiclass SegFormer MiT-B1 with shallow full-resolution refinement for fine spatial detail.
- `multiclass-segformer-mit-b2-refinement-config.yaml`: the otherwise identical refinement experiment using the larger MiT-B2 encoder.
- `multiclass-segformer-config.yaml`: the multiclass SegFormer MiT-B5 experiment with independently tunable patching, loss, and optimization settings.
- `multiclass-segformer-mit-b3-geometry-config.yaml`: multiclass SegFormer MiT-B3 with geometry-aware cross-entropy; its optional full-image validation is preconfigured for the three smallest images.
- `config_segformer_mit_b3.yaml`: binary SegFormer MiT-B3 experiment.

## Data

The normal dataset layout is:

```text
data/
├── FIVES/
│   ├── Original/
│   └── Ground truth/
├── images/
├── loci_masks/
├── inoculum_masks/
├── join_masks/             # optional, sparse join-region annotations
└── image_splits.csv
```

Images and masks are matched by filename stem. For binary training, `segmentation.target` selects either `loci` or `inoculum`; only the corresponding mask directory is used. Multiclass training uses only complete, dimension-matched image, loci-mask, and inoculum-mask sets. Incomplete or mismatched required sets are excluded from training, qualitative evaluation, and test evaluation with named warnings. Multiclass masks are composed in memory as background `0`, loci `1`, and inoculum `2`; inoculum wins where source masks overlap. Optional join masks are matched by stem and do not make an image incomplete when absent. The B2 refinement experiment enables them with:

```yaml
join_masks:
  enabled: true
  masks_dir: data/join_masks
  merge_with_loci: true
```

With `merge_with_loci: true`, join pixels are added to loci before inoculum precedence is applied. Other multiclass configs can opt in with the same section. Dimension-mismatched optional masks are named in a warning and ignored for that image.

For an ablation run that must not use join annotations during training but must retain comparable join-only test metrics and overlays, use:

```yaml
join_masks:
  enabled: false
  masks_dir: data/join_masks
  merge_with_loci: false
  evaluation_enabled: true
```

In this mode, training discovery, patch filtering, targets, loss, validation, and ordinary test metrics use only the loci and inoculum masks. Test evaluation still loads optional join masks, reports `join_pixels`, `dice_join`, and `iou_join` against an evaluation-only join target, and draws the red join boundary. Use the same evaluation config with `--config multiclass-config.yaml` when evaluating both the join-trained and ablation checkpoints so their ordinary and join-only metrics use identical targets.

The split CSV must contain `filename,split` columns. It may include forward-looking rows for images or masks that are not available yet; those rows are validated, named in a warning, and ignored until a complete usable pair exists. Every currently usable pair must still be assigned to exactly one `train`, `validation`/`val`, or `test` split, and each split must contain at least one usable pair. Invalid labels, conflicting duplicate assignments, and usable pairs missing from the CSV remain errors.

Optional FIVES retinal-vessel data can be added to training with `data.use_fives: true`. FIVES files are matched by stem between `data/FIVES/Original/` and `data/FIVES/Ground truth/`; incomplete or dimension-mismatched data fails fast. FIVES never enters fungal source splits, validation, testing, inference, or qualitative evaluation. In multiclass training, vessel pixels are loci class `1` and all other pixels are background class `0`.

## Training and splits

```bash
python -m src.train --config config.yaml
python -m src.train --config multiclass-config.yaml
python -m src.train --config multiclass-segformer-mit-b1-refinement-config.yaml
python -m src.train --config multiclass-segformer-mit-b2-refinement-config.yaml
python -m src.train --config multiclass-segformer-config.yaml
python -m src.train --config multiclass-segformer-mit-b3-geometry-config.yaml
python -m src.train --config config_segformer_mit_b3.yaml
```

Training:

1. discovers image/mask sets;
2. groups all patches by their original source image;
3. applies the configured split strategy;
4. regenerates randomized training patch records each epoch;
5. trains and validates the configured model, loss, optimizer, and scheduler;
6. saves checkpoints and metrics; and
7. runs CSV test evaluation and qualitative checkpoint comparison when enabled.

`train.seed` makes patch plans and Albumentations reproducible. Each DataLoader worker receives a distinct deterministic augmentation stream derived from that seed, so worker processes do not repeat one another's transform sequence.

Supported split modes are `csv`, `train_val`, and `kfold`. The default is `csv`. `kfold` creates grouped folds from original images and has no held-out test split. `train_val` uses `split.val_source_ids` for a single manual split.

All operational settings are in YAML. Read the selected configuration for its patch geometry, foreground filtering, augmentation probabilities, batch size, training duration, loss weights, optimizer rates, scheduler settings, and enabled post-training workflows. These experiment values are intentionally not duplicated in the documentation.

Detailed patch behavior is documented in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

Scaled-context training can enable `patching.train.scaled_context.containment_filter` to remove a smaller scaled crop when a larger retained crop covers at least the configured fraction of its source area. `threshold` controls that fraction, and `preserve_normal_patches: true` protects scale-1 patches so high-resolution supervision is retained. Filtering is per source image and does not affect validation or inference unless separately enabled in their phase settings. `src.patching.explain` reports how many patches this filter removes.

Models that expose an encoder can configure separate `optimizer.encoder_lr` and `optimizer.decoder_lr` values to fine-tune the pretrained encoder independently from the decoder and segmentation head. Maintained multiclass experiments use this setup. A mapping-valued `scheduler.min_lr` supplies matching `encoder` and `decoder` floors. The decoder group includes every trainable non-encoder component, such as shallow refinement branches and the segmentation head; exact rates, floors, and scheduler parameters belong in the selected YAML. Other configs can continue using scalar `optimizer.lr` and `scheduler.min_lr`.

The MiT-B1/B2 refinement model retains the standard SegFormer multiscale decoder at quarter resolution and replaces its final direct upsampling head with two lightweight convolutional fusion stages. A shallow input branch supplies half- and full-resolution features so the final logits can retain local morphology and connectivity cues. Its channel widths are configured under `model.shallow_channels`, `model.refine_half_channels`, and `model.refine_full_channels`.

Validation has two independently configured paths. `validation.fast` controls the patch-level pass used every epoch. `foreground_only: true` applies the shared `patching.mask_threshold` and `patching.min_foreground_pixels` rules, including the union of loci and inoculum in multiclass mode and join pixels when `join_masks.merge_with_loci` is enabled. Its `overlap` overrides training overlap for validation only; `overlap: 0` makes the nominal stride equal to the patch size. The deterministic grid still adds a final edge-covering position, so a small right/bottom overlap can remain when an image dimension is not divisible by the patch size.

`validation.full_image` controls the slower stitched pass. When `enabled: true`, it reconstructs each selected source on the normal deterministic grid, averages overlapping probabilities, and applies the binary threshold or multiclass `argmax` at the configured `interval_epochs`. `selection: smallest_area` with positive `max_images` selects a deterministic area-sorted subset (ties use source ID); `selection: all` uses every validation source. `val_dice_cldice_per_image` combines full-image foreground Dice and hard clDice using `validation.full_image.monitor`; multiclass clDice is loci-specific. A per-image training or scheduler monitor requires full-image validation enabled at an interval of one. The maintained multiclass experiment YAMLs currently disable this expensive pass and monitor `val_dice_per_patch` for faster experimentation. Training still checks all batch losses, components, and assembled epoch metrics for finite values before scheduler stepping and checkpointing.

To inspect the four FIVES patches for one image without training:

```bash
python -m src.visualize_fives_patches --config config.yaml
python -m src.visualize_fives_patches --config config.yaml --image 60_D.png
```

The command writes a numbered source overlay and the four vessel-mask overlays under `outputs/<project>/fives_patch_visualization/`. See [FIVES_REMOVAL.md](FIVES_REMOVAL.md) if this optional training support should later be removed completely.

## Run outputs

Runs are written to `runs/<project>_<timestamp>/` and include:

- an effective `config.yaml` containing the training date and only settings relevant to the selected pipeline;
- `split_manifest.csv` and `.json`;
- `cv_summary.json`, `fold_metrics.csv`, and `epoch_metrics.csv`; the epoch CSV is refreshed after every completed epoch and the fold CSV after every completed fold;
- one `fold_<n>/` directory per split, containing `best.pt`, `last.pt`, optional interval-best checkpoints, metric history, `patch_distribution.json` (including a separate FIVES summary and the full-image-validation source selection), TensorBoard logs, and checkpoint manifests; fold metric and manifest CSV/JSON files are refreshed after every completed epoch; and
- `test-evaluation/` when a CSV test split is evaluated.
- `val-train-set-evaluation/` when train/validation evaluation is run manually; it contains `val_train_set_metrics.csv`, `summary.json`, and split-specific overlays under `overlays/train/` and `overlays/validation/`.

Binary and multiclass runs use different metric fields. Binary metrics include Dice, IoU, precision, recall, clDice, and foreground fraction. Multiclass metrics include per-class scores, foreground-macro Dice/IoU, loci-only clDice, the stitched `dice_cldice_per_image` monitor, and mask-overlap diagnostics. When join masks are enabled for training, full-image validation, test evaluation, train/validation evaluation, and qualitative crop evaluation also report `dice_join`, `iou_join`, and `join_pixels` for loci recovery inside effective join regions. Test evaluation also reports these fields when only `join_masks.evaluation_enabled` is true. Images with no join mask or no effective join pixels receive blank (`null` in JSON) join scores and are excluded from join-score means. Patch-level epoch metrics and loss-component diagnostics are weighted by patch count, so a smaller final batch cannot bias the epoch result. Precision and recall score a one-sided empty prediction/target as `0.0` and a two-sided empty pair as `1.0`. A foreground class absent from both prediction and target is excluded from that sample’s multiclass macro; a sample with no predicted or target foreground receives a perfect macro score. Differential-learning-rate runs additionally record `encoder_lr` and `decoder_lr`; `lr` remains a compatibility alias for `decoder_lr`.

Fold-level and aggregate validation fields describe one consistent best checkpoint epoch. `cv_summary.json` records `segmentation_mode` and named `mask_dirs`; binary runs additionally record their target and single `mask_dir`, while multiclass runs leave those single-target fields null.

## Test evaluation

When test evaluation is enabled for a CSV split, training evaluates each test image with the selected `best.pt`. To evaluate a checkpoint later:

```bash
python -m src.inference.test_evaluation \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt
```

The command loads the run's saved config by default. Use `--config` for a checkpoint without a neighboring run config and `--output` to choose another output directory.

For binary runs, the configured inference threshold is reported and the sweep is controlled by `test_evaluation.threshold_start`, `test_evaluation.threshold_stop`, and `test_evaluation.threshold_step`. It writes `test_metrics.csv`, `threshold_metrics.csv`, masks, overlays, summary JSON, and one plot per binary metric. Test overlays compare predictions with ground truth: separate colors identify ground-truth-only pixels, prediction-only pixels, and correct overlap, with a legend in the bottom-right corner. Multiclass overlays provide those distinctions for loci and inoculum and additionally identify wrong-class overlap. When an image has an optional join mask, its boundary is drawn in red so the annotation remains visible without hiding the prediction/ground-truth result inside it. Multiclass evaluation uses `argmax`, does not sweep thresholds, and writes `multiclass_metrics.png` instead. `test_metrics.csv` and `summary.json` include join-region Dice/IoU and `num_join_images` when the feature is enabled. Only currently complete, dimension-matched test pairs are evaluated.

## Train and validation evaluation

To evaluate a run checkpoint on its train and validation images:

```bash
python -m src.inference.val_train_set_eval \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt
```

The saved run config is loaded automatically. The output CSV contains per-image rows plus `train_mean`, `validation_mean`, and `train_validation_mean` rows. Only overlays are written, under separate `overlays/train/` and `overlays/validation/` folders; masks, probability maps, and threshold plots are not produced. CSV, manual train/validation, and k-fold configurations are supported.

## Inference

For a single image or a non-recursive directory:

```bash
python -m src.inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/images \
  --output outputs/inference
```

Inference tiles each image with the deterministic patch grid, averages overlapping probabilities, and writes masks plus overlays. Binary masks contain `0`/`255`; multiclass masks contain class IDs `0`, `1`, and `2`. With `inference.save_probabilities: true`, binary probability maps are saved as `*_prob.png`; multiclass loci and inoculum maps are saved separately.

For recursive mask-only inference that preserves the complete folder structure:

```bash
python -m src.inference.recursive_masks \
  --config runs/<multiclass-run>/config.yaml \
  --checkpoint runs/<multiclass-run>/fold_0/best.pt \
  --input /path/to/fung-all-images
```

The output directory is created beside the input with `_masks` appended to its name; the example writes to `/path/to/fung-all-images_masks/`. Its root contains the effective `config.yaml` used for inference, including the original `training_date` when the run config provides one, so model and preprocessing settings remain associated with the results. Input files matching `*_mask.png` are skipped case-insensitively. Relative subdirectories and source stems are preserved. If the output directory already exists, matching outputs and `config.yaml` are overwritten and unrelated files are left in place. Inference stops before model loading if two source files in one directory share a stem and would overwrite the same outputs.

Multiclass inference writes:

- `*_mask.png`, with class IDs `0` background, `1` loci, and `2` inoculum;
- `*_loci.png`, as a binary `0`/`255` mask; and
- `*_inoculum.png`, as a binary `0`/`255` mask.

With `inference.save_probabilities: true`, multiclass configs also write `*_prob_loci.png` and `*_prob_inoculum.png`. These are grayscale probability maps where `0` represents probability `0.0` and `255` represents probability `1.0`. The workflow does not copy original images or write overlays. Binary configs write `*_mask.png` and a binary mask for the configured `loci` or `inoculum` target.

The older recursive binary workflow writes masks next to source images:

```bash
python -m src.inference.in_folder \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/images
```

Generated `*_mask.png` files are skipped on later runs. This legacy entrypoint is intended for binary models.

For images collected from other papers:

```bash
python -m src.inference.other_test_data_evaluation \
  --config config.yaml \
  --model runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/other-test-data
```

The script searches recursively, ignores its results directory, and writes masks and overlays under `<input>/results/` while preserving the source folder structure. `--model`, `--model-path`, and `--checkpoint` are aliases; `--input` and `--output` are optional.

## Qualitative evaluation

```bash
python -m src.inference.qualitative_evaluation \
  --run-dir runs/fungi_segmentation_<timestamp>
```

By default, qualitative evaluation uses its configured split from the training image and mask directories. It reads every checkpoint listed in each fold's `checkpoint_manifest.csv`, selects a reproducible crop with the configured foreground range, and writes:

- comparison grids under `qualitative_evaluation/grids/`;
- for multiclass runs, class-ID crop masks, optional loci/inoculum probability maps, and `summary.json`;
- `eval_metrics.csv` and `selected_crops.csv`; and
- for `kfold`, one global-best checkpoint comparison per fold under `fold_comparison_grids/` with `fold_comparison_metrics.csv`.

Use `--data-root` for an alternate labeled dataset. Binary roots contain `images/` and `masks/`; multiclass roots contain `images/`, `loci_masks/`, and `inoculum_masks/`. The same settings can be overridden with `--output-dir`, `--crop-patch-grid`, foreground-ratio flags, `--selection-seed`, `--threshold`, `--device`, and `--max-checkpoints`.

Foreground-ratio selection uses the configured threshold for binary source masks and any non-background class ID for composed multiclass masks.

Training runs this automatically when `qualitative_evaluation.enabled` is true.

## Models and losses

The model factory currently supports:

- `unetplusplus`, `unetplusplus_resnet18`, `unetplusplus_resnet34`, and `unetplusplus_resnet50`;
- `segformer`, `segformer_mit_b3`, and `segformer_mit_b5`; and
- `deeplabv3_resnet50` and `fcn_resnet50` baselines.

U-Net++ decoder normalization accepts `batchnorm`, `instancenorm`, `layernorm`, or `identity`; attention can be enabled with `decoder_attention_type: scse`. The SegFormer and torchvision models use their own architecture-specific normalization.

`multiclass-config.yaml`, `multiclass-segformer-config.yaml`, and `multiclass-segformer-mit-b3-geometry-config.yaml` use the same multiclass pipeline but are independent experiments. They may tune patching, augmentations, losses, optimization, scheduling, and training duration separately; consult the selected YAML for exact values.

Binary loss names include BCE aliases, `bce_dice`, `bce_dice_cldice`, `tversky`, `cldice`, `soft_cldice`, and `tversky_soft_cldice`. The clDice loss implementations use differentiable soft skeletonization, while reported hard-mask clDice skeletonizes binary masks to convergence. Multiclass training supports `multiclass_ce_dice_loci_cldice`: cross-entropy plus foreground Dice plus loci soft-clDice. It also supports `multiclass_geometry_ce_dice_loci_cldice`, which replaces ordinary cross-entropy with target-derived pixel weighting. After augmentation, loci center depth is normalized by the nearest skeleton radius, optional multi-scale closing emphasizes narrow annotated background separators, inoculum is excluded from separator weighting, and all raw weights remain at least `1`. The weighted cross-entropy is divided by the sum of its weights; Dice and loci soft-clDice are unchanged.

## Patch inspection

For binary target-specific patch diagnostics:

```bash
python -m src.patching.explain --config config.yaml --target loci --epoch 1
python -m src.patching.explain \
  --config config.yaml \
  --target loci \
  --image "data/images/example.tif" \
  --epoch 1
```

The command reports candidate, kept, discarded, normal, and scaled-context patch counts, scale statistics, and split/source summaries. With `--image`, it writes an overlay under `outputs/<project>/patching_explain/`. The maintained patching behavior and limitations of this binary diagnostic are described in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

## Development layout

See [DESCRIPTION.md](DESCRIPTION.md) for the module map, data flow, output ownership, and maintenance boundaries.
