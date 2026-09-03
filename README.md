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

- `multiclass-config.yaml`: three classes—background, loci, and inoculum—with U-Net++/ResNet50.
- `multiclass-segformer-mit-b1-refinement-config.yaml`: multiclass SegFormer MiT-B1 with shallow full-resolution refinement for fine spatial detail.
- `multiclass-segformer-mit-b2-refinement-config.yaml`: the otherwise identical refinement experiment using the larger MiT-B2 encoder.
- `multiclass-segformer-config.yaml`: the multiclass SegFormer MiT-B5 experiment with independently tunable patching, loss, and optimization settings.
- `multiclass-segformer-mit-b3-geometry-config.yaml`: obsolete geometry-weighting experiment retained temporarily for compatibility; do not use it for new runs.
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

Resume an interrupted cross-validation run at the beginning of its first unfinished fold with:

```bash
python -m src.train --resume-run runs/<run-directory>
```

`--config` and `--resume-run` are mutually exclusive. Resume always reloads the run's saved `config.yaml`, verifies the recomputed split manifest exactly, preserves completed folds and the append-only log, removes partial artifacts for the first unfinished and later folds, and restarts that fold at epoch 1 with a fresh model/optimizer/scheduler. `run_state.json` is the completion source for new runs; legacy runs infer the contiguous completed prefix from `fold_metrics.csv`. Resume events and removed paths are recorded in `resume_history.json`.

Training:

1. discovers image/mask sets;
2. groups all patches by their original source image;
3. applies the configured split strategy;
4. builds larger static training regions once, then regenerates final-crop records each epoch;
5. trains and validates the configured model, loss, optimizer, and scheduler;
6. saves checkpoints and metrics; and
7. runs CSV test evaluation and qualitative checkpoint comparison when enabled.

`train.seed` makes patch plans and Albumentations reproducible, with a stable fold-specific seed so an unfinished fold can restart without replaying earlier folds. Each DataLoader worker receives a distinct deterministic augmentation stream. `data.num_workers`, `persistent_workers`, and `prefetch_factor` apply to one training DataLoader worker pool even when tqdm is enabled. With `data.train_patch_cache.enabled: true` (the default), the run decodes the union of fungal sources used for training across all folds once and writes unaugmented uint8 RGB and mask regions to a temporary run-level NumPy memmap. Each static region has side length `patch_size + overlap` (768 for a 512 patch and 256 overlap). Every epoch selects an independent deterministic `patch_size x patch_size` crop inside each interior region, while leading and trailing edge anchors remain fixed. Workers read only those final crops before augmentation, normalization, geometry weighting, and iteration metadata. The cache is deleted after training; FIVES remains on its direct path. Stitched full-image validation runs in the main process.

Supported split modes are `csv`, `csv_kfold`, `train_val`, and `kfold`. The default is `csv`. `csv_kfold` keeps the CSV `test` sources fixed and creates seeded grouped folds from the combined CSV `train` and `validation` sources; every pooled source appears in validation exactly once, so fold sizes can differ by one when the pool is not divisible by `cv.n_splits`. Plain `kfold` uses every discovered source in grouped folds and has no held-out test split. `train_val` uses `split.val_source_ids` for a single manual split.

All operational settings are in YAML. Read the selected configuration for its patch geometry, foreground filtering, augmentation probabilities, batch size, training duration, loss weights, optimizer rates, scheduler settings, and enabled post-training workflows. These experiment values are intentionally not duplicated in the documentation.

Detailed patch behavior is documented in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

Scaled-context training can enable `patching.train.scaled_context.containment_filter` to remove a smaller scaled crop when a larger retained crop covers at least the configured fraction of its source area. `threshold` controls that fraction, and `preserve_normal_patches: true` protects scale-1 patches so high-resolution supervision is retained. Filtering is per source image and does not affect validation or inference unless separately enabled in their phase settings. `src.patching.explain` reports how many patches this filter removes. Static training caching and training scaled context are mutually exclusive because a fixed `patch_size + overlap` region cannot provide arbitrary source-scale context; configuration loading rejects the combination.

`patching.train.background_only` optionally retains strictly zero-foreground final training crops. Foreground counting and filtering occur after the epoch crop is selected from its static region; foreground elsewhere in the larger region does not qualify the final crop. With `enabled: true`, `percentage_of_foreground` is calculated independently for each image after foreground and containment filtering and rounded up. Every image with an available background candidate retains at least one such patch even when the calculated quota is zero, and the result is capped by availability. Checked-in experiment YAMLs enable the feature but keep their percentage independently tunable; read the selected YAML for its active value. Selection is deterministic for the configured seed and epoch; validation, test evaluation, and inference are unchanged.

Models that expose an encoder can configure separate `optimizer.encoder_lr` and `optimizer.decoder_lr` values to fine-tune the pretrained encoder independently from the decoder and segmentation head. Maintained multiclass experiments use this setup. A mapping-valued `scheduler.min_lr` supplies matching `encoder` and `decoder` floors. The decoder group includes every trainable non-encoder component, such as shallow refinement branches and the segmentation head; exact rates, floors, and scheduler parameters belong in the selected YAML. Other configs can continue using scalar `optimizer.lr` and `scheduler.min_lr`.

The MiT-B1/B2 refinement model retains the standard SegFormer multiscale decoder at quarter resolution and replaces its final direct upsampling head with two lightweight convolutional fusion stages. A shallow input branch supplies half- and full-resolution features so the final logits can retain local morphology and connectivity cues. Its channel widths are configured under `model.shallow_channels`, `model.refine_half_channels`, and `model.refine_full_channels`.

Training uses stitched full-image validation only. `validation.start_epoch` sets the first validation epoch; earlier epochs are training-only and do not update validation-selected checkpoints or metric-based schedulers. Legacy `validation.fast` mappings are accepted and discarded when old configurations are loaded.

`validation.full_image` reconstructs each selected source with a fixed stride of `patching.patch_size // 2` and runs transformed image and target patches in groups of `batch_size`. Each batch is inferred once: the configured loss is evaluated from its logits and targets, and those same logits are converted to probabilities for stitching. Overlapping probabilities are averaged before the binary threshold or multiclass `argmax`. The deterministic grid appends a final edge-covering position, so dimensions not divisible by patch size can add limited right or bottom overlap. `selection: smallest_area` with positive `max_images` selects a deterministic area-sorted subset (ties use source ID); `selection: all` uses every validation source. Training requires `enabled: true` and `interval_epochs: 1` so checkpoint and scheduler monitors are available on every validation epoch.

Validation reports stitched Dice, IoU, and fast Zhang hard-clDice. The immutable target skeleton is cached per source for the fold, while each stitched prediction is skeletonized once per validation epoch and reused by every hard-clDice-based monitor. For multiclass validation, `val_dice_per_image` is the arithmetic mean of stitched loci and inoculum Dice; `best_current.pt` selects `0.7 * val_dice_per_image + 0.3 * val_cldice_loci_per_image`, while `best_val_loss.pt` minimizes patch-count-weighted validation loss. When `validation.full_image.soft_cldice_foreground_only: true`, CE and Dice still use every overlapping patch, while differentiable soft-clDice runs only for samples whose target contains loci/class 1 (or positive target pixels in binary mode). Skipped samples contribute zero and normalization remains over the original batch. Training soft-clDice is unmasked.

New multiclass folds store `best_current.pt`, pure-Dice `best_dice.pt`, `best_low_cldice.pt` (0.9 Dice + 0.1 loci hard-clDice), `best_inoculum_compensated.pt` (0.3 loci Dice + 0.5 inoculum Dice + 0.2 loci hard-clDice), and `best_val_loss.pt`; binary folds store combined `best_current.pt`, `best_dice.pt`, and `best_val_loss.pt`. Multiclass CE, Dice, and soft-clDice logging reuses tensors from the actual loss calculation instead of recomputing the loss. `train.compute_hard_cldice_metrics: false` disables the expensive per-training-batch CPU skeleton diagnostic while retaining train Dice, IoU, precision, recall, and loss-derived components. This setting does not disable validation hard-clDice, gradients, or validation-only checkpoint selection. Checked-in `ReduceLROnPlateau` schedulers also minimize `val_loss` and remain idle before `validation.start_epoch`. No validation-selected checkpoint exists until the first validation pass completes. Training checks all assembled metrics for finite values before scheduler stepping and checkpointing.

CSV test evaluation is stitched with a fixed 50% overlap: its stride is always `patching.patch_size // 2`, regardless of the configured `patching.stride` or `patching.overlap`. Validation uses the same fixed geometry; training, ordinary inference, and recursive inference retain their configured stride.

### Soft-clDice iteration analysis

With static caching and a Soft-clDice loss, `loss.static_patch_iterations.enabled: true` computes the exact required count once from each cached region. For multiclass training it uses the effective loci target after optional join merging and inoculum precedence. `margin_iterations` is then added and the result is rounded upward to `round_up_to`; the active experiment uses 10 for both. The cached value is a conservative bound for every epoch crop from that region and avoids applying a full-image outlier count to all of its patches. Empty regions receive the configured margin after rounding.

The existing CSV workflow remains available as an explicit per-image override. To build it, first build the exact crisp-mask table from
all top-level loci masks:

```bash
python -m src.build_soft_cldice_iteration_map \
  --mask-dir data/loci_masks \
  --output-csv data/loci_soft_cldice_required_iterations.csv
```

Then add an absolute safety margin for thicker or otherwise imperfect model
predictions:

```bash
python -m src.add_soft_cldice_iteration_margin \
  --input-csv data/loci_soft_cldice_required_iterations.csv \
  --output-csv data/loci_soft_cldice_training_iterations.csv \
  --margin-iterations 10 \
  --round-up-to 10
```

The first CSV contains one row per mask with
`mask_filename,mask_stem,width,height,foreground_pixels,required_iterations`.
The second preserves those columns and adds `margin_iterations`,
`round_up_to`, and `training_iterations`. After adding the margin and
applying the optional minimum, values are rounded upward to multiples of
`--round-up-to` (default `10`) so mixed batches contain fewer distinct
skeletonization groups. An optional `--maximum-iterations` cap is applied
last and can therefore produce a capped value outside the bucket multiples.
The exact count is the last useful four-neighbour erosion for the crisp
full-resolution ground truth; margining and bucketing remain separate from
mask scanning so they can be changed cheaply.

A configured table overrides automatic static-region values:

```yaml
loss:
  iterations: 30
  iterations_csv: data/loci_soft_cldice_training_iterations.csv
```

When `iterations_csv` is set, training requires a valid row for every
discovered fungal target/loci mask and propagates its `training_iterations`
to all patches from that source. Within a mixed batch, samples are grouped by
iteration count and skeletonized separately, so only the corresponding groups
pay for high counts. The fixed `loss.iterations` remains the fallback for optional FIVES patches, cache-disabled training, or disabled automatic calculation. With static caching enabled, leave `iterations_csv: null` to use automatic static-region counts. These CSVs are dataset-derived local metadata and should follow the
same privacy/versioning policy as the masks from which they were computed.

For a more detailed convergence, timing, and visual analysis:

To choose the differentiable soft-clDice erosion count from all loci masks:

```bash
python -m src.analyze_soft_skeleton_iterations \
  --mask-dir data/loci_masks \
  --min-iterations 30 \
  --max-iterations 150 \
  --iteration-step 10 \
  --visual-iterations 30,50,70,90,110,130,150 \
  --output-dir outputs/soft-skeleton-iteration-analysis
```
To reproduce the selected thick-mask outlier analysis with full-resolution
binary artifacts:

```bash
python -m src.analyze_soft_skeleton_iterations \
  --mask-dir data/loci_masks \
  --mask-name "IN7 F MEA 48h x500 1.png" \
  --mask-name "IN7 F YNA 48h x500 3.png" \
  --mask-name "IN7 J SNA 48h x500 2.png" \
  --min-iterations 100 \
  --max-iterations 214 \
  --iteration-step 50 \
  --visual-iterations 100,150,200,214 \
  --save-full-resolution \
  --output-dir outputs/soft-skeleton-outlier-analysis
```

Repeated `--mask-name` values select exact top-level filenames and fail if any
are missing. With `--save-full-resolution`, each selected mask gets
`full_resolution/<stem>/ground_truth.png` plus one
`soft_skeleton_<iterations>.png` for every tested setting.


The analyzer applies the production soft skeletonizer to every readable
top-level mask at original resolution. It uses haloed tiles so the result is
identical to full-image execution through the configured maximum iteration
without requiring a full-resolution float tensor. CUDA is selected when
available; use `--device cpu` to override it.

`aggregate_iterations.csv` is the primary decision table. Select the smallest
iteration with an acceptable worst-image `minimum_capture_vs_max_iteration`
and measured kernel time. The default summary recommendation requires every
mask to capture at least 99.9% of its iteration-150 skeleton. Check
`summary.json.all_images_complete_at_max_tested` before accepting that
recommendation: `maximum_exact_required_iterations` uses the maximum
city-block foreground radius to state exactly when crisp target skeletons have
finished changing. If it exceeds 150, the sweep measures convergence toward 150
rather than a fully converged skeleton.

The exact bound describes the uncropped full masks. Training operates on
configured patches (and may rescale or augment them), so patch boundaries can
lower the effective radius while prediction errors may increase it. The
two-stage CSV pipeline therefore keeps the exact conservative source value and
the prediction-error margin auditable.

`per_image_iterations.csv` identifies outlier masks, new skeleton pixels at
each tested step, residual eroded foreground, measured kernel time, and Dice
against the maximum tested iteration. Per-image contact sheets under
`visuals/` show the mask, Zhang reference, and selected soft skeletons.

The aggregate benchmark uses synchronized CUDA events (or `perf_counter` on
CPU) around only the production skeletonization recurrence, cumulatively
through each tested iteration over all masks. It excludes transfers, metric
reduction, Zhang skeletonization, output I/O, and autograd; actual training
forward/backward wall time will therefore be higher.

Zhang raw pixel overlap is not an iteration target because Zhang thinning and
the differentiable morphological skeleton can choose different centerline
pixels; the two-pixel tolerant F1 in `summary.json` and the contact sheets are
spatial sanity checks. Ordinary repeated binary erosion alone is insufficient
because it shows the shrinking mask, not the accumulated skeleton used by
soft-clDice.

### Hard-clDice CPU/CUDA benchmark

Use two dimension-matched binary masks to compare paper-reference hard-clDice skeletonization on CPU and CUDA:

```bash
python -m src.benchmark_cldice \
  --prediction-mask path/to/prediction.png \
  --target-mask path/to/target.png \
  --output-dir outputs/cldice-benchmark/sample \
  --skeletonizer paper \
  --repeats 1
```

Binary masks are thresholded at `127` by default. For class-index masks, use
`--foreground-value 1` to benchmark loci only. `--skeletonizer paper` is the
only supported selection: CPU uses `skimage.morphology.skeletonize(method="zhang")`,
while CUDA uses an equivalent PyTorch implementation only for benchmark
comparison. Explicit test evaluation and diagnostic hard-clDice workflows use the CPU
scikit-image implementation; epoch validation does not compute it. Each device is synchronized around the
measured skeletonization and clDice calculation; input transfer time is
reported separately. `summary.json` and `timings.csv` contain algorithms,
timings, speedup, clDice difference, skeleton Dice/IoU, differing-pixel counts,
and CUDA peak allocation. The output directory also contains CPU and GPU
skeleton PNGs and comparison overlays where white is shared, red is present
only in the first named skeleton, cyan only in the second, and black in neither.

Use a representative full-resolution pair: tiny masks can be slower on CUDA because launch overhead dominates. Avoid running the benchmark beside training when measuring speed, because both processes would contend for the GPU. The default single repeat is intentional because the current CPU implementation can take many minutes on large masks.

To benchmark hard-clDice on multiclass-style foreground patches, select a
seeded subset of foreground patches and process them in validation-sized
batches:

```bash
python -m src.benchmark_cldice_patches \
  --prediction-mask path/to/prediction.png \
  --target-mask path/to/target.png \
  --output-dir outputs/cldice-patch-benchmark/sample \
  --patch-size 512 \
  --stride 256 \
  --num-patches 50 \
  --batch-size 8 \
  --seed 42 \
  --repeats 1
```

Only patches containing foreground in the prediction or target are eligible.
The benchmark saves per-patch metrics plus contact sheets for the inputs,
CPU/GPU skeletons, and skeleton-overlap comparisons.

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
- `run_state.json`, optional `resume_history.json`, `cv_summary.json`, `fold_metrics.csv`, and `epoch_metrics.csv`; epoch rows include `train_duration_seconds` (cache build plus training) and `validation_duration_seconds` (stitching plus metrics, excluding checkpoint writes);
- one `fold_<n>/` directory per split, containing monitor-specific best checkpoints (five for new multiclass folds and three for binary folds, as described above), optional `last.pt` and interval-best checkpoints according to `train.save_last_checkpoint` and `train.best_interval_checkpoint`, metric history, `patch_distribution.json` (including a separate FIVES summary and the full-image-validation source selection), TensorBoard logs, and checkpoint manifests; fold metric and manifest CSV/JSON files are refreshed after every completed epoch; and
- `test-evaluation/` when a CSV-backed test split is evaluated. Checkpoint selections that resolve to the same epoch share one inference under `fold_<n>/epoch_<epoch>/` (or `epoch_<epoch>/` for one fold). Root `checkpoint_comparison.csv` directly compares every fold/checkpoint pair, while `monitor_comparison_summary.csv` reports each monitor’s cross-fold means, population standard deviations, and contributing-fold counts. Per-fold `checkpoint_comparison.csv` files are also written. The existing root `fold_metrics.csv`, `summary.csv`, and `summary.json` remain the cross-fold summary for `best_current.pt`.
- `val-train-set-evaluation/` when train/validation evaluation is run manually; it contains `val_train_set_metrics.csv`, `summary.json`, and split-specific overlays under `overlays/train/` and `overlays/validation/`.

Binary and multiclass runs use different metric fields. Binary metrics include Dice, IoU, precision, recall, clDice, and foreground fraction. Multiclass training/validation metrics include per-class scores, foreground-macro Dice/IoU, and mask-overlap diagnostics. Zhang hard-clDice remains available in explicit test/benchmark workflows, not epoch validation. When join masks are enabled for training, full-image validation, test evaluation, train/validation evaluation, and qualitative crop evaluation also report `dice_join`, `iou_join`, and `join_pixels` for loci recovery inside effective join regions. Test evaluation also reports these fields when only `join_masks.evaluation_enabled` is true. Images with no join mask or no effective join pixels receive blank (`null` in JSON) join scores and are excluded from join-score means. Training patch-level metrics and loss-component diagnostics are weighted by patch count, so a smaller final batch cannot bias the epoch result. Precision and recall score a one-sided empty prediction/target as `0.0` and a two-sided empty pair as `1.0`. A foreground class absent from both prediction and target is excluded from that sample’s multiclass macro; a sample with no predicted or target foreground receives a perfect macro score. Differential-learning-rate runs additionally record `encoder_lr` and `decoder_lr`; `lr` remains a compatibility alias for `decoder_lr`.

Fold-level and aggregate validation fields describe one consistent best checkpoint epoch. `cv_summary.json` records `segmentation_mode` and named `mask_dirs`; binary runs additionally record their target and single `mask_dir`, while multiclass runs leave those single-target fields null.

## Test evaluation

When test evaluation is enabled for a CSV-backed split, training evaluates every available validation-monitor checkpoint selection on held-out CSV test sources. Selections from the same fold and epoch share one inference and state-oriented artifact directory. Comparison CSVs retain one row per monitor mapping and add `evaluation_id`, `canonical_evaluated_checkpoint`, `shared_evaluation`, and `matching_checkpoint_names`; distinct selected epochs are still evaluated separately. The legacy run-level test fields and cross-fold `fold_metrics.csv`/`summary.csv`/`summary.json` continue to describe `best_current.pt`. To evaluate a checkpoint later:

```bash
python -m src.inference.test_evaluation \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best_current.pt
```

The command loads the run's saved config by default. Use `--config` for a checkpoint without a neighboring run config and `--output` to choose another output directory.

For binary runs, the configured inference threshold is reported and the sweep is controlled by `test_evaluation.threshold_start`, `test_evaluation.threshold_stop`, and `test_evaluation.threshold_step`. It writes `test_metrics.csv`, `threshold_metrics.csv`, masks, overlays, summary JSON, and one plot per binary metric. Test overlays compare predictions with ground truth: separate colors identify ground-truth-only pixels, prediction-only pixels, and correct overlap, with a legend in the bottom-right corner. Multiclass overlays provide those distinctions for loci and inoculum and additionally identify wrong-class overlap. When an image has an optional join mask, its boundary is drawn in red so the annotation remains visible without hiding the prediction/ground-truth result inside it. Multiclass evaluation uses `argmax`, does not sweep thresholds, and writes `multiclass_metrics.png` instead. `test_metrics.csv` and `summary.json` include join-region Dice/IoU and `num_join_images` when the feature is enabled. Only currently complete, dimension-matched test pairs are evaluated.

Test prediction always uses a stride of `patching.patch_size // 2` so overlapping probabilities are averaged with 50% nominal overlap. This is independent of the stride used during training and ordinary inference; validation uses the same half-patch stride.

## Train and validation evaluation

To evaluate a run checkpoint on its train and validation images:

```bash
python -m src.inference.val_train_set_eval \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best_current.pt
```

The saved run config is loaded automatically. The output CSV contains per-image rows plus `train_mean`, `validation_mean`, and `train_validation_mean` rows. Only overlays are written, under separate `overlays/train/` and `overlays/validation/` folders; masks, probability maps, and threshold plots are not produced. CSV, manual train/validation, and k-fold configurations are supported.

## Inference

For a single image or a non-recursive directory:

```bash
python -m src.inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best_current.pt \
  --input data/images \
  --output outputs/inference
```

Inference tiles each image with the deterministic patch grid, averages overlapping probabilities, and writes masks plus overlays. Binary masks contain `0`/`255`; multiclass masks contain class IDs `0`, `1`, and `2`. With `inference.save_probabilities: true`, binary probability maps are saved as `*_prob.png`; multiclass loci and inoculum maps are saved separately.

For recursive mask-only inference that preserves the complete folder structure:

```bash
python -m src.inference.recursive_masks \
  --config runs/<multiclass-run>/config.yaml \
  --checkpoint runs/<multiclass-run>/fold_0/best_current.pt \
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
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best_current.pt \
  --input data/images
```

Generated `*_mask.png` files are skipped on later runs. This legacy entrypoint is intended for binary models.

For images collected from other papers:

```bash
python -m src.inference.other_test_data_evaluation \
  --config config.yaml \
  --model runs/fungi_segmentation_<timestamp>/fold_0/best_current.pt \
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

`multiclass-config.yaml` and `multiclass-segformer-config.yaml` are independent multiclass experiments. `multiclass-segformer-mit-b3-geometry-config.yaml` is an obsolete compatibility experiment pending removal. They may tune patching, augmentations, losses, optimization, scheduling, and training duration separately; consult the selected YAML for exact values.

Binary loss names include BCE aliases, `bce_dice`, `bce_dice_cldice`, `tversky`, `cldice`, `soft_cldice`, and `tversky_soft_cldice`. The clDice loss implementations use the paper's differentiable soft skeletonization, while reported hard-mask clDice uses CPU `skimage.morphology.skeletonize(method="zhang")`. Multiclass training supports `multiclass_ce_dice_loci_cldice`: cross-entropy plus foreground Dice plus loci soft-clDice. The `multiclass_geometry_ce_dice_loci_cldice` geometry-weighting loss is obsolete, retained only for compatibility with existing configs and checkpoints, and should not be selected for new experiments. It is planned for later removal.

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
