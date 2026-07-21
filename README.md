# Fungi Segmentation

Modular PyTorch pipeline for patch-based semantic segmentation of fungal networks in RGB microscopy or macroscopy images. It supports binary segmentation of one target at a time and multiclass segmentation of loci plus inoculum.

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.train --config config.yaml
```

The checked-in `config.yaml` is the main binary experiment: U-Net++ with a ResNet34 encoder, 512-pixel patches, 256-pixel stride, and the CSV train/validation/test split. Other ready-to-use configurations are:

- `multiclass-config.yaml`: three classes—background, loci, and inoculum—with U-Net++/ResNet34.
- `multiclass-segformer-config.yaml`: the same multiclass experiment with SegFormer MiT-B3.
- `config_segformer_mit_b3.yaml`: legacy binary SegFormer MiT-B3 experiment.
- `config-small-run.yaml`: legacy single-mask layout for small compatibility runs.

## Data

The normal dataset layout is:

```text
data/
├── images/
├── loci_masks/
├── inoculum_masks/
└── image_splits.csv
```

Images and masks are matched by filename stem. For binary training, `segmentation.target` selects either `loci` or `inoculum`; only the corresponding mask directory is used. The CSV must contain `filename,split` columns and assign every discovered image to `train`, `validation`/`val`, or `test`.

Multiclass training requires matching image, loci-mask, and inoculum-mask stems. Incomplete sets and dimension-mismatched sets are excluded during discovery. Masks are composed in memory as background `0`, loci `1`, and inoculum `2`; inoculum wins where the two source masks overlap.

## Training and splits

```bash
python -m src.train --config config.yaml
python -m src.train --config multiclass-config.yaml
python -m src.train --config multiclass-segformer-config.yaml
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

Supported split modes are `csv`, `train_val`, and `kfold`. The default is `csv`. `kfold` creates grouped folds from original images and has no held-out test split. `train_val` uses `split.val_source_ids` for a single manual split.

All important settings are in YAML. The current default `config.yaml` uses:

- `patching.patch_size: 512`, `patching.stride: 256`;
- foreground filtering at `min_foreground_pixels: 128`;
- epoch-specific random grid offsets and scaled-context patches;
- `train.monitor: val_dice_macro_resolution`; and
- automatic full-image validation every 10 epochs.

Detailed patch behavior is documented in [PATCHING_DESCRIPTION.md](PATCHING_DESCRIPTION.md).

## Run outputs

Runs are written to `runs/<project>_<timestamp>/` and include:

- the merged `config.yaml`;
- `split_manifest.csv` and `.json`;
- `cv_summary.json`, `fold_metrics.csv`, and `epoch_metrics.csv`;
- one `fold_<n>/` directory per split, containing `best.pt`, `last.pt`, optional interval-best checkpoints, metric history, `patch_distribution.json`, TensorBoard logs, and checkpoint manifests; and
- `test-evaluation/` when a CSV test split is evaluated.

Binary and multiclass runs use different metric fields. Binary metrics include Dice, IoU, precision, recall, clDice, and foreground fraction. Multiclass metrics include per-class scores, foreground-macro Dice/IoU, loci-only clDice, and mask-overlap diagnostics.

## Test evaluation

Training evaluates each CSV test image with the selected `best.pt`. To evaluate a checkpoint later:

```bash
python -m src.test_evaluation \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt
```

The command loads the run's saved config by default. Use `--config` for a checkpoint without a neighboring run config and `--output` to choose another output directory.

For binary runs, the configured inference threshold is reported and the default sweep covers `0.50` through `1.00` in `0.01` increments. It writes `test_metrics.csv`, `threshold_metrics.csv`, masks, overlays, summary JSON, and one plot per binary metric. Multiclass evaluation uses `argmax`, does not sweep thresholds, and writes `multiclass_metrics.png` instead.

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

For recursive binary inference that writes masks next to the source images:

```bash
python -m src.in_folder_inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/images
```

Generated `*_mask.png` files are skipped on later runs. This entrypoint is intended for binary models.

For images collected from other papers:

```bash
python -m src.other_test_data_evaluation \
  --config config.yaml \
  --model runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/other-test-data
```

The script searches recursively, ignores its results directory, and writes masks and overlays under `<input>/results/` while preserving the source folder structure. `--model`, `--model-path`, and `--checkpoint` are aliases; `--input` and `--output` are optional.

## Qualitative evaluation

```bash
python -m src.qualitative_evaluation \
  --run-dir runs/fungi_segmentation_<timestamp>
```

By default, qualitative evaluation uses the configured split (normally CSV `test`) from `data/images` and the active mask target. It reads every checkpoint listed in each fold's `checkpoint_manifest.csv`, selects a reproducible crop with the configured foreground range, and writes:

- comparison grids under `qualitative_evaluation/grids/`;
- crop masks/probabilities when enabled;
- `eval_metrics.csv` and `selected_crops.csv`; and
- for `kfold`, one global-best checkpoint comparison per fold under `fold_comparison_grids/` with `fold_comparison_metrics.csv`.

Use `--data-root` for an alternate labeled dataset. Binary roots contain `images/` and `masks/`; multiclass roots contain `images/`, `loci_masks/`, and `inoculum_masks/`. The same settings can be overridden with `--output-dir`, `--crop-patch-grid`, foreground-ratio flags, `--selection-seed`, `--threshold`, `--device`, and `--max-checkpoints`.

Training runs this automatically when `qualitative_evaluation.enabled` is true.

## Models and losses

The model factory currently supports:

- `unetplusplus`, `unetplusplus_resnet18`, `unetplusplus_resnet34`, and `unetplusplus_resnet50`;
- `segformer` and `segformer_mit_b3`; and
- `deeplabv3_resnet50` and `fcn_resnet50` baselines.

U-Net++ decoder normalization accepts `batchnorm`, `instancenorm`, `layernorm`, or `identity`; attention can be enabled with `decoder_attention_type: scse`. The SegFormer and torchvision models use their own architecture-specific normalization.

`multiclass-config.yaml` and `multiclass-segformer-config.yaml` intentionally share the same data, patches, augmentations, loss, optimizer, scheduler, monitors, test evaluation, and qualitative evaluation. They differ only in project identity and model architecture, making their results directly comparable.

Binary losses include BCE, Dice, Tversky, clDice, soft-clDice, and their configured combinations. Multiclass training uses `multiclass_ce_dice_loci_cldice`: cross-entropy plus foreground Dice plus loci soft-clDice.

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
