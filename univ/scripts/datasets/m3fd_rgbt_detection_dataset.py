"""M3FD RGB-IR detection dataset for downstream UNIV experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class M3FDRGBTDetectionDataset(Dataset):
    def __init__(self, root: str, split: str = "train", imgsz: int = 224, modality: str = "rgb-ir") -> None:
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.modality = modality
        self.visible_dir = self._find_dir([f"images/{split}/visible", f"images/{split}/rgb", f"{split}/visible", f"{split}/rgb", "visible", "rgb"])
        self.infrared_dir = self._find_dir([f"images/{split}/infrared", f"images/{split}/ir", f"{split}/infrared", f"{split}/ir", "infrared", "ir"])
        self.label_dir = self._find_dir([f"labels/{split}", f"{split}/labels", "labels"], required=False)
        self.samples = self._pair_samples()
        if not self.samples:
            raise FileNotFoundError(f"No paired visible/infrared M3FD samples found under {self.root}")

    def _find_dir(self, candidates: List[str], required: bool = True) -> Path | None:
        for c in candidates:
            p = self.root / c
            if p.is_dir():
                return p
        if required:
            raise FileNotFoundError(f"Could not find one of {candidates} under {self.root}")
        return None

    def _pair_samples(self) -> List[Tuple[Path, Path, Path | None]]:
        ir_by_stem = {p.stem: p for p in self.infrared_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        samples = []
        for vp in sorted(p for p in self.visible_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            ip = ir_by_stem.get(vp.stem)
            if ip is None:
                continue
            lp = self.label_dir / f"{vp.stem}.txt" if self.label_dir else None
            samples.append((vp, ip, lp if lp and lp.exists() else None))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.imgsz, self.imgsz))
        return TF.to_tensor(img)

    def _load_labels(self, path: Path | None) -> torch.Tensor:
        if path is None:
            return torch.zeros((0, 5), dtype=torch.float32)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
        return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        vp, ip, lp = self.samples[idx]
        return {"visible": self._load_image(vp), "infrared": self._load_image(ip), "targets": self._load_labels(lp), "path": str(vp)}
