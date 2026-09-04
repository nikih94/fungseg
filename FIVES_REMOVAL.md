# Removing optional FIVES training support

FIVES support is isolated and disabled by default. To remove it completely:

1. Remove `data.use_fives` from `DEFAULT_CONFIG` in `src/utils/config.py` and from maintained YAML configurations that declare it.
2. Remove the FIVES imports, record loading/logging, `FivesPatchDataset` (including its geometry-weight-builder pass-through), `combine_training_datasets`, FIVES patch diagnostics, and FIVES fold count from `src/train.py`. Restore the static-cached (or direct, when disabled) fungal dataset as the direct argument to the training loader; do not remove `src/data/patch_cache.py`, whose training and validation caches are independent of FIVES.
3. Delete `src/data/fives.py`, `src/visualize_fives_patches.py`, and `tests/test_fives.py`.
4. Remove FIVES sections and file references from `README.md`, `DESCRIPTION.md`, `PATCHING_DESCRIPTION.md`, and `AGENTS.md`, then delete this file.
5. Confirm that `rg -n "FIVES|fives|use_fives" --glob '!data/**' .` has no feature references and run the full unit test suite.
