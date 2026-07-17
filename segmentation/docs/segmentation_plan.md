# Segmentation Plan

1. Validate MFNet and FMB alignment and class IDs with `check_seg_dataset.py`.
2. Train RGB-only, IR-only, and RGB-IR early-fusion U-Net baselines.
3. Train the UNIV-original encoder adapter under the same evaluation protocol.
4. Report pixel accuracy, per-class IoU, mean IoU, memory, and reproducibility details.
5. Add PSMAF components only after the baseline table is complete.
