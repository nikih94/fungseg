# Agent Guide for `fungseg`

This file is the repository-level working agreement for coding agents. It applies to the entire repository unless a more specific `AGENTS.md` is added in a subdirectory.

## Mission and source of truth

`fungseg` is a config-driven PyTorch project for patch-based fungal-network segmentation. It supports:

- binary segmentation of one configured target (`loci` or `inoculum`); and
- multiclass segmentation with background `0`, loci `1`, and inoculum `2`.

When documentation and implementation disagree, inspect the code, active YAML configurations, and tests first. Then update the documentation in the same change. Do not preserve a stale behavior in docs merely because it appears in an older document.

## Repository structure

```text
.
├── AGENTS.md                         # agent instructions and maintenance rules
├── README.md                         # user quickstart and operational guide
├── DESCRIPTION.md                    # architecture and module map
├── PATCHING_DESCRIPTION.md           # detailed patch-generation behavior
├── FIVES_REMOVAL.md                  # removal checklist for optional FIVES training support
├── config.yaml                       # current binary experiment
├── multiclass-config.yaml            # multiclass U-Net++/ResNet34 experiment
├── multiclass-segformer-mit-b1-refinement-config.yaml # multiclass MiT-B1 refinement experiment
├── multiclass-segformer-mit-b2-refinement-config.yaml # multiclass MiT-B2 refinement experiment
├── multiclass-segformer-config.yaml  # multiclass SegFormer MiT-B5 experiment
├── multiclass-segformer-mit-b3-geometry-config.yaml # multiclass MiT-B3 geometry-loss experiment
├── config_segformer_mit_b3.yaml      # binary SegFormer MiT-B3 experiment
├── requirements.txt                  # runtime dependencies
├── data/                             # local datasets and split metadata
├── src/                              # application code
│   ├── inference/                    # inference, recursive prediction, and evaluation workflows
│   ├── models/segformer_refinement.py # MiT-B1/B2 full-resolution refinement model
│   └── ...                           # data, engine, models, patching, metrics, and utilities
├── tests/                            # unit and integration tests
├── runs/                             # generated training runs
├── outputs/                          # generated inference/diagnostic outputs
├── extract-mask.py                   # standalone mask-extraction utility
└── venv/                             # local virtual environment, if present
```

Archives, historical output folders, checkpoints, generated images, `__pycache__`, and the local virtual environment are artifacts rather than application modules. Do not edit or commit generated artifacts unless the task explicitly requires it.

## Data organization

The default dataset is organized as follows:

```text
data/
├── FIVES/                  # optional training-only retinal images and masks
│   ├── Original/
│   └── Ground truth/
├── images/                 # source microscopy/macroscopy images
├── loci_masks/             # binary loci masks, matched by filename stem
├── inoculum_masks/         # binary inoculum masks, matched by filename stem
├── join_masks/             # optional sparse join-region masks, matched by filename stem
├── image_splits.csv        # filename,split assignments for train/validation/test
├── other-test-data/        # external paper image collections for inference
├── small-test/             # auxiliary local test data
└── test/                   # auxiliary local test data
```

Rules:

- Training discovery is top-level within the configured image and mask directories; matching is by filename stem.
- Binary mode uses the directory selected by `segmentation.target` and `paths.mask_dirs`.
- Multiclass mode uses only complete, dimension-matched image/loci-mask/inoculum-mask sets. Training and evaluation exclude incomplete or mismatched required sets with named warnings. Configured join masks are optional per image; valid masks can be merged into loci for training, or loaded only by CSV test evaluation for join metrics and red overlay boundaries. Absent masks leave targets unchanged and dimension-mismatched optional masks are ignored with warnings.
- `data/image_splits.csv` must contain `filename,split`. Forward-looking rows without a currently usable pair are validated, named in a warning, and ignored. Every currently usable pair must be assigned to exactly one non-empty `train`, `validation`/`val`, or `test` split.
- Split membership is assigned to original images, never to individual patches. Never introduce patch-level leakage between train and validation/test.
- `data/other-test-data/` is recursively processed by `src.inference.other_test_data_evaluation`; its generated results belong under its results directory and must not be treated as training data.
- `data/FIVES/` is optional auxiliary training data controlled by `data.use_fives`; it never participates in fungal splits, validation, or test evaluation.
- `data/small-test/` and `data/test/` are auxiliary local data locations, not inputs to the default `config.yaml` run unless a configuration explicitly points to them.

Do not commit private datasets, large archives, checkpoints, or generated masks unless the repository explicitly requires them.

## Code organization

### Entrypoints

- `src/train.py`: discovery, split construction, loaders, model/loss/optimizer/scheduler construction, fold execution, test evaluation, and optional qualitative evaluation.
- `src/inference/__main__.py`: single-image or non-recursive directory inference with overlapping-patch stitching.
- `src/inference/recursive_masks.py`: recursive binary/multiclass mask-only inference into a mirrored sibling directory.
- `src/inference/in_folder.py`: legacy recursive binary inference that writes masks next to source images.
- `src/inference/other_test_data_evaluation.py`: recursive inference for external paper image collections.
- `src/inference/test_evaluation.py`: CSV test-split evaluation and threshold/class metric artifacts.
- `src/inference/val_train_set_eval.py`: train/validation evaluation for a supplied best checkpoint, with aggregate CSV metrics and split-specific overlays.
- `src/inference/qualitative_evaluation.py`: checkpoint comparison on selected labeled crops.
- `src/visualize_fives_patches.py`: one-image diagnostic for optional FIVES center-patch geometry.

### Core packages

- `src/inference/`: shared image loading, patch prediction, stitching, output conversion, and inference/evaluation entrypoints.
- `src/data/`: discovery, source-level split logic, lazy datasets, transforms, and patch diagnostics.
- `src/patching/`: `OriginalImageRecord`, `PatchRecord`, deterministic edge-covering grids, training randomization, scaled context, resampling, and foreground filtering.
- `src/models/`: model factory, SegFormer full-resolution refinement, output normalization, and decoder normalization helpers.
- `src/losses/`: binary and multiclass loss implementations plus the loss factory.
- `src/metrics/`: segmentation metrics and loss-component diagnostics.
- `src/engine/`: training/validation loop, stitched full-image validation, checkpointing, metric export, and TensorBoard logging.
- `src/optim/` and `src/schedulers/`: configurable optimizer and scheduler factories.
- `src/utils/`: config merging/compatibility, checkpoint I/O, serialization, logging, and reproducibility helpers.

Keep responsibilities within these boundaries. Add behavior to the appropriate package and expose it through an entrypoint only when it is an actual user workflow. Avoid putting model-specific logic in the trainer, split logic in datasets, or hard-coded paths in reusable modules.

## Development invariants

1. Configuration is the source of truth. Add new user-tunable behavior to YAML and `src/utils/config.py` defaults rather than hard-coding it in an entrypoint.
2. Preserve source-image grouping. Any new sampler, patch transform, or validation path must retain source IDs and prevent data leakage.
3. Keep binary and multiclass semantics explicit. Binary models use one sigmoid output and thresholding; multiclass models use three class logits, softmax, and argmax. Do not silently apply binary threshold logic to multiclass outputs.
4. Preserve deterministic geometry. Fast patch validation must cover image edges according to its configured overlap; stitched full-image validation, test evaluation, and inference must cover edges and average overlapping predictions consistently.
5. Use the existing factories for models, losses, optimizers, and schedulers. When adding a supported option, update the relevant factory, configuration example, tests, and docs together.
6. Treat each checked-in experiment YAML as an independent source of truth. Tests may enforce shared pipeline semantics such as multiclass class IDs, output shape, and monitor availability, but must not require tunable patching, augmentation, loss, optimizer, scheduler, or training values to match another experiment. Architecture guides document option behavior; exact experiment values belong in the active YAML.
7. Keep masks and predictions shape-safe. Check image/mask dimensions at discovery or evaluation boundaries and preserve class IDs or binary encoding when saving outputs.
8. Prefer small, composable functions with type hints and `pathlib.Path`. Keep entrypoints thin and make reusable behavior testable without requiring a full training run.
9. Avoid unnecessary dependencies. If a dependency is required at runtime, add it to `requirements.txt` and document why.
10. Do not change user data, runs, outputs, checkpoints, or environment files as part of a code change unless that mutation is explicitly requested.

## Standard workflow for agents

Before editing:

1. Read this file and the relevant section of `README.md`, `DESCRIPTION.md`, or `PATCHING_DESCRIPTION.md`.
2. Inspect the active configuration and the implementation that owns the behavior.
3. Search for callers, tests, output names, and documentation references before renaming or removing anything.
4. Check the working tree and preserve unrelated user changes.

While editing:

- Make the smallest coherent change that satisfies the task.
- Reuse existing helpers and conventions before adding parallel implementations.
- Update tests for changed behavior, especially for split logic, patch geometry, output encoding, configuration compatibility, and multiclass behavior.
- Keep generated files out of the patch.

After editing:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Also run focused tests or lightweight module checks relevant to the change. Do not launch a full training run unless requested or necessary for verification. Review the final diff for stale names, hard-coded run timestamps, undocumented files, and accidental generated artifacts.

## Documentation synchronization policy

Documentation is part of the implementation contract. Every code, config, data-layout, CLI, output, or repository-structure change must include the corresponding documentation update in the same change.

Use these ownership rules:

- `README.md`: user-facing commands, supported workflows, data setup, configuration behavior, model/loss choices, and output usage.
- `DESCRIPTION.md`: architecture, module responsibilities, data flow, ownership boundaries, and run-artifact structure.
- `PATCHING_DESCRIPTION.md`: patch size/stride/overlap, edge coverage, offsets, scaled context, resampling, foreground filtering, phase behavior, and patch diagnostics.
- `FIVES_REMOVAL.md`: complete removal checklist for optional FIVES support; update it when that support's integration points change.
- `AGENTS.md`: repository tree, data organization, development invariants, workflow, and documentation-maintenance rules.

Mandatory synchronization checklist:

- If a file, directory, module, entrypoint, config, or output location is added, removed, renamed, or repurposed, update the repository tree and relevant descriptions in `AGENTS.md` and `DESCRIPTION.md`.
- If a CLI flag, config key, model, loss, split mode, output artifact, or data requirement changes, update `README.md` and the owning architecture/patching guide.
- If patch construction or filtering changes, update `PATCHING_DESCRIPTION.md` and its references in `README.md`/`DESCRIPTION.md`.
- Remove obsolete commands, filenames, hard-coded run examples, and claims about behavior that no longer exists.
- Keep links and command examples pointing to files that actually exist. The canonical patching filename is `PATCHING_DESCRIPTION.md`.
- Do not mirror tunable YAML values in architecture guides unless an example is explicitly illustrative; changing only an experiment parameter does not require a documentation edit.
- Do not finish a structural change while `AGENTS.md` still describes the old repository layout. The structure section must be updated in the same change.

At the end of every task, search the documentation for the old name or behavior and verify that the documented tree, commands, and outputs match the current repository.

## Handoff expectations

Final reports should state:

- what changed and which files were touched;
- what verification was run and its result;
- any known limitations or intentionally unchanged compatibility paths; and
- any user action still required.
