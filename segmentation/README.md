# Segmentation Module

This directory is the semantic segmentation development module for the PSMAF-Net research project.

PSMAF-Net stands for **Pseudo-Semantic guided Multi-scale Adaptive Fusion Network**.

Current goal:
- Build RGB-IR semantic segmentation baselines.
- Keep this module separated from the existing detection experiments.
- Add UNIV-original + segmentation head as an important baseline.

Do not store checkpoints, logs, datasets, or training outputs in Git.

## Included baselines

* **U-Net RGB-only** (`unet_rgb`)
* **U-Net IR-only** (`unet_ir`)
* **U-Net RGB-IR early fusion** (`unet_early_fusion`, channel concatenation)
* **UNIV-original + segmentation head** (`univ_seg`), which preserves the
  existing UNIV implementation and attaches a decoder in this directory.

## Dataset layout

The paired loader expects aligned files with matching stems. Split files contain
one stem (or relative filename) per line:

```text
dataset/
├── rgb/       # three-channel images
├── ir/        # single-channel infrared images
├── masks/     # integer class-index masks
└── splits/{train,val}.txt
```

Copy one of the YAML files in `configs/`, set the dataset root, class count and
image size, then run a short experiment with:

```bash
python segmentation/scripts/train.py \
  --config segmentation/configs/unet_early_fusion.yaml \
  --output segmentation/runs/early_fusion
python segmentation/scripts/evaluate.py \
  --config segmentation/configs/unet_early_fusion.yaml \
  --checkpoint segmentation/runs/early_fusion/best.pth
```

The UNIV configuration accepts an upstream checkpoint in
`model.checkpoint`. When it is omitted, the architecture is initialized from
scratch (useful only for smoke testing). UNIV uses a 224×224 input by default,
matching its fixed positional embedding. No dataset or weights are downloaded
by these scripts.
