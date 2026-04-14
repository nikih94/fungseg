# Fungi Segmentation

Modular PyTorch scaffold for binary semantic segmentation of fungi images using grouped k-fold cross-validation on image-level sources and patch-based training.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Place source images in `data/images` and their binary masks in `data/masks`. Images and masks are matched by filename stem.

## Train

```bash
python -m src.train --config config.yaml
```

This scans the dataset, builds patch records in memory, runs grouped cross-validation by original image stem, and saves fold checkpoints under `runs/<project>_<timestamp>/fold_<n>/`.
Each training launch creates a timestamped run folder under `runs/` containing the merged config, fold checkpoints, TensorBoard logs, and CSV/JSON metrics.

## Inference

```bash
python -m src.inference \
  --config config.yaml \
  --checkpoint runs/fungi_segmentation_<timestamp>/fold_0/best.pt \
  --input data/images \
  --output outputs/inference
```

Inference uses the same patch size and overlap as training, averages probabilities in overlapping regions, and writes binary masks, green overlay previews, and optional probability maps.

The current default model in `config.yaml` is `Unet++` with an ImageNet-pretrained `resnet50` encoder. The model factory also supports `unetplusplus_resnet18` and `unetplusplus_resnet34`.
