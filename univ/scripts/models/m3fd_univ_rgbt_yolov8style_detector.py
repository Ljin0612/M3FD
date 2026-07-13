"""UNIV-M3FD RGB-IR detector with shared UNIV encoder and YOLOv8-style heads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from univ.models.backbone.mcmae.models_convmae import convmae_convvit_base_patch16
except ImportError:  # pragma: no cover
    from models.backbone.mcmae.models_convmae import convmae_convvit_base_patch16


@dataclass
class LoadReport:
    weights_exists: bool = False
    weights_loaded: bool = False
    checkpoint_key: str = "student"
    loaded_keys: int = 0
    skipped_keys: int = 0
    missing_keys: int = 0
    unexpected_keys: int = 0


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOv8StyleHead(nn.Module):
    def __init__(self, channels: Tuple[int, int, int], nc: int) -> None:
        super().__init__()
        self.nc = nc
        self.heads = nn.ModuleList(
            nn.Sequential(ConvBNAct(c, c), nn.Conv2d(c, 4 + 1 + nc, 1)) for c in channels
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(feat).permute(0, 2, 3, 1).contiguous() for head, feat in zip(self.heads, feats)]


class M3FDUNIVRGBTYOLOv8StyleDetector(nn.Module):
    def __init__(self, nc: int = 6, imgsz: int = 224, shared_encoder: bool = True, fusion: str = "concat") -> None:
        super().__init__()
        if not shared_encoder:
            raise NotImplementedError("First downstream adapter version supports shared_encoder=True only.")
        self.nc = nc
        self.imgsz = imgsz
        self.shared_encoder = shared_encoder
        self.fusion = fusion
        self.encoder = convmae_convvit_base_patch16()
        self.fuse14 = nn.Conv2d(768 * 2 if fusion == "concat" else 768, 256, 1)
        self.reduce28 = ConvBNAct(256, 128, 1)
        self.reduce7 = ConvBNAct(256, 256, 3)
        self.head = YOLOv8StyleHead((128, 256, 256), nc)
        self.load_report = LoadReport()

    def _encode_feature(self, x: torch.Tensor) -> torch.Tensor:
        # Use zero mask ratio to preserve all patches and reuse the UNIV encoder path.
        latent, _ = self.encoder(x, mask_ratio=0.0, return_last_attention=False)
        b, n, c = latent.shape
        side = int(n ** 0.5)
        return latent.transpose(1, 2).reshape(b, c, side, side)

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> List[torch.Tensor]:
        vf = self._encode_feature(visible)
        irf = self._encode_feature(infrared)
        if self.fusion == "weighted_sum":
            fused = 0.5 * vf + 0.5 * irf
        else:
            fused = torch.cat([vf, irf], dim=1)
        p4 = self.fuse14(fused)
        p3 = self.reduce28(F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False))
        p5 = self.reduce7(F.max_pool2d(p4, 2))
        return self.head([p3, p4, p5])

    def freeze_backbone(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False


def load_univ_checkpoint(model: M3FDUNIVRGBTYOLOv8StyleDetector, path: str | None, checkpoint_key: str = "student") -> LoadReport:
    report = LoadReport(checkpoint_key=checkpoint_key)
    if not path:
        print("No --univ-weights provided; using randomly initialized UNIV encoder.")
        model.load_report = report
        return report
    report.weights_exists = Path(path).is_file()
    if not report.weights_exists:
        print(f"WARNING: UNIV weights not found: {path}")
        model.load_report = report
        return report
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get(checkpoint_key, ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint key {checkpoint_key!r} did not contain a state dict")
    model_state = model.encoder.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in state.items():
        kk = k.removeprefix("module.").removeprefix("backbone.").removeprefix("encoder.")
        if kk in model_state and tuple(model_state[kk].shape) == tuple(v.shape):
            loadable[kk] = v
        else:
            skipped.append(k)
    incompatible = model.encoder.load_state_dict(loadable, strict=False)
    report.loaded_keys = len(loadable)
    report.skipped_keys = len(skipped)
    report.missing_keys = len(incompatible.missing_keys)
    report.unexpected_keys = len(incompatible.unexpected_keys)
    report.weights_loaded = report.loaded_keys > 0
    model.load_report = report
    print(
        f"UNIV checkpoint_key={checkpoint_key} loaded_keys={report.loaded_keys} "
        f"skipped_keys={report.skipped_keys} missing_keys={report.missing_keys} "
        f"unexpected_keys={report.unexpected_keys}"
    )
    return report
