# Repository Description

This repository contains a config-driven PyTorch segmentation pipeline. The main workflow discovers source images and masks, creates image-grouped patch records, trains a selected model, evaluates stitched full images, and stores auditable run artifacts.

## Repository map

- `config.yaml`: current binary U-Net++/ResNet34 experiment.
- `multiclass-config.yaml`: loci/inoculum multiclass U-Net++/ResNet34 experiment.
- `multiclass-segformer-mit-b1-refinement-config.yaml`: loci/inoculum multiclass SegFormer MiT-B1 full-resolution-refinement experiment.
- `multiclass-segformer-mit-b2-refinement-config.yaml`: the matching full-resolution-refinement experiment with a MiT-B2 encoder.
- `multiclass-segformer-config.yaml`: loci/inoculum multiclass SegFormer MiT-B5 experiment.
- `multiclass-segformer-mit-b3-geometry-config.yaml`: loci/inoculum multiclass SegFormer MiT-B3 geometry-loss experiment.
- `config_segformer_mit_b3.yaml`: binary SegFormer MiT-B3 experiment.
- `FIVES_REMOVAL.md`: removal checklist for optional FIVES training support.
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
- `src/build_soft_cldice_iteration_map.py`: computes the exact crisp-target soft-skeleton completion count for every top-level loci mask and writes the raw per-mask CSV.
- `src/add_soft_cldice_iteration_margin.py`: converts the raw iteration CSV into training values by applying configurable absolute safety margins, upward iteration bucketing, and optional bounds.
- `src/analyze_soft_skeleton_iterations.py`: sweeps the production differentiable skeletonizer over full-resolution binary masks in exact haloed tiles, reports per-image and aggregate convergence, computes the crisp-target city-block radius bound, and writes Zhang-reference visual contact sheets.
- `src/benchmark_cldice.py`: benchmarks paper-reference Zhang hard-clDice using scikit-image on CPU and an equivalent lookup-table implementation on CUDA, with timing, skeleton-equivalence, and visual overlap artifacts; production evaluation remains CPU-based.
- `src/benchmark_cldice_patches.py`: benchmarks the same hard-clDice path on a seeded foreground-only patch subset with configurable patch and batch sizes.
- `src/inference/__main__.py`: predicts on one image or a non-recursive directory using stitched overlapping patches.
- `src/inference/recursive_masks.py`: recursively writes combined and class-specific masks, plus optional class probability maps, into a mirrored sibling directory.
- `src/inference/in_folder.py`: legacy recursive binary inference that writes masks next to source images.
- `src/inference/other_test_data_evaluation.py`: recursive inference for external paper image collections, preserving their folder structure in a results directory.
- `src/inference/test_evaluation.py`: evaluates a checkpoint on the CSV `test` split and writes full-image metrics and artifacts.
- `src/inference/val_train_set_eval.py`: train/validation checkpoint evaluation with split-specific overlays and aggregate metrics.
- `src/inference/qualitative_evaluation.py`: compares manifest checkpoints on selected labeled crops.
- `src/visualize_fives_patches.py`: draws the four fixed center patches used for one FIVES training image.

The `src/inference/core.py` module owns image loading, deterministic patch prediction, probability stitching, mask conversion, and overlays shared by these entrypoints. `src/inference/__init__.py` re-exports its public helpers, and `src/inference/__main__.py` provides the `python -m src.inference` command.

### `src/data/`

- `discovery.py`: matches images and masks by filename stem. Multiclass discovery also checks that every required mask exists and has the image dimensions.
- `fives.py`: strictly discovers optional FIVES pairs, builds their centered 2×2 records, and maps vessel masks to binary foreground or multiclass loci.
- `folds.py`: implements CSV, CSV-held-out grouped cross-validation, manual train/validation, and plain grouped k-fold source splits.
- `dataset.py`: lazy patch loading, binary/multiclass mask composition, resizing, normalization, Albumentations transforms, and optional post-transform target-weight construction.
- `sampling.py`: patch-distribution summaries saved for each training fold.
- `soft_cldice_iterations.py`: exact crisp-mask iteration geometry plus strict loading and source-record mapping for adjusted training CSVs.

The dataset layer keeps source-image grouping separate from patch-level samples, so train/validation boundaries cannot be created by splitting patches from the same original image.

### `src/patching/`

- `core.py`: `OriginalImageRecord` and `PatchRecord`, deterministic edge-covering grids, epoch-specific training offsets, scaled-context crops, source-crop containment filtering, deterministic per-source background-only retention, mask resampling, and foreground filtering.
- `explain.py`: binary target patch diagnostics, per-source background counts, and optional color-coded overlay generation.

The full patching contract is maintained in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

### `src/models/`

- `factory.py`: builds SMP U-Net++, SMP SegFormer, the MiT-B1/B2 refinement architecture, or torchvision FCN/DeepLabV3 models from config.
- `segformer_refinement.py`: combines a standard MiT-B1 or MiT-B2 encoder and quarter-resolution SegFormer decoder with shallow half/full-resolution features and convolutional fusion blocks.
- `wrappers.py`: normalizes SMP and torchvision outputs through `extract_logits()`.
- `norms.py`: channel-wise 2D LayerNorm used for U-Net++ decoder normalization.

### `src/losses/`

- `factory.py`: maps configured loss names to implementations.
- `combined.py`: binary BCE/Dice, Tversky, clDice, soft-clDice, combined losses, and multiclass ordinary or geometry-aware CE/Dice/loci-clDice losses, whose components are promoted to FP32 for mixed-precision stability and accept per-sample skeleton iteration counts.
- `geometry.py`: builds target-derived loci contour-depth and optional background-separator weight maps after augmentation using Euclidean distance transforms, skeletonization, and morphology.

### `src/metrics/`

- `segmentation.py`: binary and multiclass Dice, IoU, precision, recall, shared differentiable soft-clDice math with iteration-grouped batches, paper-reference CPU Zhang hard-mask skeletonization through scikit-image, and clDice evaluation from generated or precomputed skeletons.
- `loss_components.py`: diagnostics for loss terms and component scores written with training metrics; clDice diagnostics reuse the loss implementation.

### `src/engine/`

- `trainer.py`: fold training, fast patch validation, optional stitched full-image Dice/clDice validation, per-batch finite-value checks, sample-weighted patch diagnostics, scheduler stepping, checkpointing, TensorBoard logging, and checkpoint manifests.

### `src/optim/` and `src/schedulers/`

- `optim/factory.py` builds scalar-learning-rate optimizers or labeled encoder/decoder parameter groups; the decoder group includes every trainable non-encoder parameter, including the segmentation head.
- `schedulers/factory.py` builds the configured scheduler and resolves named per-group minimum learning rates into optimizer group order.

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
4. `folds.py` assigns source images to CSV, CSV-held-out cross-validation, manual, or grouped k-fold splits.
5. `dataset.py` loads crops lazily and applies transforms. The training seed initializes the main Albumentations pipeline, and each DataLoader worker receives a distinct deterministic derived stream.
   When `data.use_fives` is enabled, the fixed FIVES dataset is concatenated here only for training and uses the same training augmentations.
   When `loss.iterations_csv` is configured, the adjusted value mapped from each source mask is attached to every fungal patch; FIVES patches use `loss.iterations`.
6. The model, loss, optimizer, and scheduler factories build the training stack.
7. `trainer.py` begins validation at `validation.start_epoch`, groups configured per-sample soft-clDice counts so low-count samples do not run outlier recurrences, then runs fast patch validation on each validation epoch and optionally reconstructs the configured full-image sources at `validation.full_image.interval_epochs`. The stitched path batches model forwards according to `validation.full_image.batch_size`; its crop/transform loop remains in the main process rather than a DataLoader.
8. `inference/test_evaluation.py` reloads each selected `best.pt` for CSV test evaluation.
9. `inference/qualitative_evaluation.py` compares checkpoints when enabled.

### Multiclass training

Multiclass discovery makes only complete, dimension-matched image/loci-mask/inoculum-mask sets available to downstream workflows; incomplete or mismatched required sets are excluded with named warnings. Configured `join_masks` are optional per source: matching, dimension-safe masks are attached to records, absent masks leave the source usable, and invalid optional masks are ignored with named diagnostics. When training `enabled` and `merge_with_loci` are both true, the dataset unions join pixels into loci before applying inoculum precedence. With training disabled and `evaluation_enabled: true`, training and ordinary evaluation targets remain unchanged while CSV test evaluation loads join masks for join-only metrics and red overlay boundaries. Join-only scoring composes a separate loci target containing the join annotation, so it remains defined without changing ordinary metrics. For the geometry-aware loss, it derives the weight map from the augmented class-index target inside the DataLoader worker so weights and supervision stay spatially aligned. Models output three logits. Fast validation reports patch-level foreground-macro Dice/IoU; optional full-image validation batches patch inference, averages overlapping softmax probabilities before `argmax`, and additionally reports hard loci clDice. Their configured normalized weighted mean is `val_dice_cldice_per_image`. A class absent from both prediction and target is excluded from that sample’s foreground macro, while a completely foreground-empty sample scores `1.0`. Binary sigmoid thresholds and binary threshold sweeps are not used for this mode.

The U-Net++ and SegFormer multiclass YAML files, including the MiT-B1/B2 refinement and MiT-B3 geometry-loss experiments, are independent experiments using the same architecture-neutral dataset, trainer, test-evaluation, inference, and qualitative-evaluation paths. The refinement architecture keeps SegFormer context fusion at quarter resolution, then combines it with a shallow input branch at half and full resolution before classification. Its tunable patching, channel widths, augmentation, loss, optimization, scheduling, and training values come from the selected YAML.

### Inference

`src/inference/core.py` uses the same validation normalization and deterministic patch geometry as full-image validation. It averages overlapping sigmoid probabilities for binary models or softmax probabilities for multiclass models. Edge patches are padded for model input and cropped back to the valid image area before stitching.

`src/inference/recursive_masks.py` discovers source images recursively, excludes existing `*_mask.png` inputs, and mirrors relative paths under a sibling `<input-name>_masks/` directory. It saves the effective inference configuration as `<input-name>_masks/config.yaml`, preserving the run's training date when available. Multiclass prediction writes a class-ID `*_mask.png` plus binary `*_loci.png` and `*_inoculum.png`. When `inference.save_probabilities` is enabled, it also writes grayscale `*_prob_loci.png` and `*_prob_inoculum.png` maps with probabilities scaled from `[0.0, 1.0]` to `[0, 255]`. It never copies originals or writes overlays. Binary prediction writes `*_mask.png` and the configured target-specific mask when the target is loci or inoculum. Existing matching outputs and the saved configuration are overwritten without clearing the output tree, and same-directory source-stem collisions fail before model loading.

## Splits and patch ownership

Splits are always made from original image IDs, never from patch records. In CSV-backed modes, the file may pre-register future sources that do not yet have a complete usable pair; these rows are validated, reported by filename, and ignored by training, qualitative evaluation, and test evaluation. Every currently usable pair must remain assigned, and the resulting train, validation, and test groups must each be non-empty. `csv_kfold` fixes the CSV test group in every fold and distributes the combined CSV train/validation pool across seeded grouped validation folds, with every pooled source validating once. Plain k-fold mode distributes all discovered source images across grouped validation folds without a held-out test set. Manual mode creates one train/validation split from `val_source_ids`.

Training patch records are regenerated with `train.seed + epoch`. Fast validation records are deterministic, use half-patch stride, and can be limited to foreground patches while retaining source identity. They reuse the shared patch mask threshold and minimum foreground count. Optional stitched validation covers every deterministic grid position with stride `patching.patch_size // 2`, including background-only regions, for sources selected by `validation.full_image`; `smallest_area` sorts by pixel area and source ID. CSV test evaluation uses the same half-patch stride and averages overlapping probabilities; ordinary inference continues to use the configured `patching.stride`.

FIVES source IDs are namespaced with `FIVES/` and never enter fungal split construction or manifests. Their four records per image remain fixed across epochs, use scale `1.0`, and bypass fungal offsets, scaled context, and foreground filtering. Only the configured Albumentations training pipeline varies their samples.

## Run artifacts

Each run stores an effective config and split manifest at its root. The config records `training_date` in `YYYY-MM-DD` format and removes known defaults that cannot affect the selected segmentation mode, model family, loss, scheduler, or split mode. The root epoch CSV is refreshed after every completed epoch, while the fold CSV is refreshed after each completed fold and uses validation fields from that fold's best epoch. Summary metadata records the segmentation mode and named mask directories rather than describing multiclass runs as a single binary target. Each fold stores `best.pt`, optional last and interval-best checkpoints controlled by the training configuration, per-epoch history, patch-distribution diagnostics, TensorBoard data, and a checkpoint manifest recording checkpoint reason, epoch, monitor, and monitor value. Fold metric and checkpoint-manifest CSV/JSON artifacts are refreshed after every completed epoch. Epoch metrics are checked for finite values before artifacts are written, and checkpoint optimizer/scheduler state is captured after scheduler stepping.

CSV-backed test evaluation writes masks, ground-truth-versus-prediction overlays, `test_metrics.csv`, `threshold_metrics.csv`, and `summary.json`. It always uses half-patch stride (`patching.patch_size // 2`) for stitched prediction, independently of the experiment's main stride. Full-image validation uses the same half-patch stride. Multi-fold evaluation stores these artifacts under `test-evaluation/fold_<n>/` for each best checkpoint and incrementally writes root `fold_metrics.csv`, `summary.csv`, and `summary.json` containing cross-fold mean, population standard deviation, and contributing-fold counts for every available mean metric. The run-level `cv_summary.json` embeds the same aggregate payload. Each overlay has a bottom-right legend and distinct colors for ground-truth-only, prediction-only, and correct-overlap pixels. Multiclass overlays distinguish these states per foreground class, mark wrong-class overlap separately, and draw the boundary of an available join mask in red without obscuring the result inside it. Binary evaluation also writes per-metric threshold plots. Multiclass evaluation writes class metrics and `multiclass_metrics.png` without a threshold sweep. For configured join masks, full-image validation and evaluation artifacts additionally report loci Dice/IoU restricted to effective join pixels. Empty or absent join regions use `None`/blank scores and are excluded from aggregate join means.
Train/validation evaluation writes `val-train-set-evaluation/` with `val_train_set_metrics.csv`, `summary.json`, and overlays under separate `overlays/train/` and `overlays/validation/` folders; it does not write masks, probabilities, or threshold plots.

Qualitative evaluation writes grids, selected crop metadata, crop-level metrics, and optional masks/probabilities under `qualitative_evaluation/`. Binary crop foreground uses the configured source-mask threshold; multiclass crop foreground is any non-background class ID. K-fold runs additionally write the best-per-fold comparison artifacts.

## Configuration boundaries

Configuration is the source of truth for paths, segmentation mode, target/classes, optional join-mask supervision, optional FIVES training data, patch geometry, augmentations, split strategy, model, loss, optimizer, scheduler, training, inference, test evaluation, and qualitative evaluation. `loss.iterations_csv` optionally selects a strict adjusted per-mask table; null retains the fixed `loss.iterations`. Checked-in YAML files are independent experiments and are not required to share tunable values. `config.py` supplies defaults for omitted keys, removes the inherited scalar optimizer rate when a config selects differential rates, and supports old single-mask configs through `paths.masks_dir`; new configs should use `paths.mask_dirs` and an explicit segmentation target or mode. Optimizers accept either scalar `lr` or paired `encoder_lr`/`decoder_lr`; ReduceLROnPlateau accepts either scalar `min_lr` or a mapping keyed by the labeled optimizer groups.

The patch-specific options and their phase behavior belong in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md), rather than being duplicated here.

## Dependencies and tests

The runtime stack is PyTorch/torchvision, segmentation-models-pytorch, Albumentations, Pillow, NumPy, SciPy, scikit-image, PyYAML, TensorBoard, tqdm, and Matplotlib. Tests cover configuration and split compatibility, patching and multiscale crops, models, losses, metrics, optimization, DataLoader behavior, binary/multiclass inference and evaluation, qualitative evaluation, and optional FIVES support.

Run the test suite with:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```
