"""Paired RGB/infrared semantic-segmentation dataset."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedSegmentationDataset(Dataset):
    """Load aligned ``rgb``, ``ir`` and ``masks`` files listed by stem.

    A split file contains one sample stem per line. File extensions are resolved
    automatically, which keeps the loader usable for common PNG/JPEG layouts.
    """

    def __init__(self, root, split, image_size=None, augment=False, ignore_index=255):
        self.root = Path(root)
        split_path = Path(split)
        if not split_path.is_absolute():
            split_path = self.root / split_path
        self.samples = [line.strip().split()[0] for line in split_path.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
        if not self.samples:
            raise ValueError(f"No samples found in {split_path}")
        self.image_size = tuple(image_size) if image_size else None
        self.augment = augment
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.samples)

    def _resolve(self, folder, stem):
        candidate = self.root / folder / stem
        if candidate.is_file():
            return candidate
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = self.root / folder / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Cannot resolve {folder}/{stem} under {self.root}")

    def __getitem__(self, index):
        stem = self.samples[index]
        rgb = Image.open(self._resolve("rgb", stem)).convert("RGB")
        ir = Image.open(self._resolve("ir", stem)).convert("L")
        mask = Image.open(self._resolve("masks", stem))
        if rgb.size != ir.size or rgb.size != mask.size:
            raise ValueError(f"Unaligned modalities for sample {stem}")
        if self.image_size:
            size = (self.image_size[1], self.image_size[0])
            rgb = rgb.resize(size, Image.Resampling.BILINEAR)
            ir = ir.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            rgb, ir, mask = (im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in (rgb, ir, mask))
        rgb_t = torch.from_numpy(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
        ir_t = torch.from_numpy(np.asarray(ir, dtype=np.float32)[None].copy()) / 255
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return {"rgb": rgb_t, "ir": ir_t, "mask": mask_t, "id": stem}
