"""M3FD RGB-IR detection dataset for downstream UNIV experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class M3FDRGBTDetectionDataset(Dataset):
    def __init__(self, root: str, split: str = "train", imgsz: int = 224, modality: str = "rgb-ir") -> None:
        cfg = self._load_data_config(root)
        self.root = Path(cfg.get("path", cfg.get("data", root))).expanduser()
        self.split = split
        self.imgsz = imgsz
        self.modality = modality
        self.visible_dir = self._resolve_dir(cfg.get("vi", "vi"), "visible images")
        self.infrared_dir = self._resolve_dir(cfg.get("ir", "ir"), "infrared images")
        self.label_dir = self._resolve_dir(cfg.get("labels", "labels"), "labels")
        split_file = cfg.get(split, f"meta/{split}.txt")
        self.split_file = self._resolve_file(split_file, f"{split} split")
        self.samples = self._pair_samples()
        if not self.samples:
            raise FileNotFoundError(f"No paired visible/infrared/label M3FD samples found from {self.split_file} under {self.root}")

    def _load_data_config(self, root: str) -> Dict[str, Any]:
        path = Path(root).expanduser()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"path": root}

    def _resolve_dir(self, value: str, name: str) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.root / p
        if not p.is_dir():
            raise FileNotFoundError(f"Missing {name} directory for M3FD split '{self.split}': {p}")
        return p

    def _resolve_file(self, value: str, name: str) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.root / p
        if not p.is_file():
            raise FileNotFoundError(f"Missing {name} file for M3FD split '{self.split}': {p}")
        return p

    def _resolve_image(self, directory: Path, sample_id: str, name: str) -> Path:
        sid = Path(sample_id.strip())
        candidates = [directory / sid.name] if sid.suffix else [directory / f"{sid.name}{ext}" for ext in sorted(IMG_EXTS)]
        for p in candidates:
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"Missing {name} image for M3FD split '{self.split}', sample '{sample_id}': tried "
            + ", ".join(str(p) for p in candidates)
        )

    def _resolve_label(self, sample_id: str) -> Path:
        sid = Path(sample_id.strip())
        label_name = f"{sid.stem}.txt" if sid.suffix else f"{sid.name}.txt"
        lp = self.label_dir / label_name
        if not lp.is_file():
            raise FileNotFoundError(f"Missing label for M3FD split '{self.split}', sample '{sample_id}': {lp}")
        return lp

    def _pair_samples(self) -> List[Tuple[Path, Path, Path]]:
        samples = []
        for raw in self.split_file.read_text(encoding="utf-8").splitlines():
            sample_id = raw.strip()
            if not sample_id:
                continue
            vp = self._resolve_image(self.visible_dir, sample_id, "visible")
            ip = self._resolve_image(self.infrared_dir, sample_id, "infrared")
            lp = self._resolve_label(sample_id)
            samples.append((vp, ip, lp))
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
