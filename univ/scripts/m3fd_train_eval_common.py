"""Common utilities for UNIV-M3FD RGB-IR YOLOv8-style detection."""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import yaml

CLASS_NAMES = ["people", "car", "bus", "motorcycle", "lamp", "truck"]
NC = len(CLASS_NAMES)


def load_yaml(path: str | os.PathLike[str] | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def cuda_memory_string() -> str:
    if not torch.cuda.is_available():
        return "cuda=N/A"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return f"cuda_allocated_mb={allocated:.1f},cuda_reserved_mb={reserved:.1f}"


def write_kv(path: str | os.PathLike[str], values: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k}: {v}\n")


def append_csv(path: str | os.PathLike[str], row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def xywhn_to_xyxy_abs(labels: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if labels.numel() == 0:
        return labels.new_zeros((0, 5))
    cls = labels[:, 0:1]
    x, y, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    return torch.cat([cls, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], dim=1)


def collate_m3fd(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "visible": torch.stack([b["visible"] for b in batch], 0),
        "infrared": torch.stack([b["infrared"] for b in batch], 0),
        "targets": [b["targets"] for b in batch],
        "paths": [b.get("path", "") for b in batch],
    }


def simple_detection_loss(preds: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], nc: int) -> Dict[str, torch.Tensor]:
    device = preds[0].device
    obj_terms = []
    cls_terms = []
    box_terms = []
    for p in preds:
        bsz, h, w, _ = p.shape
        obj = p[..., 4]
        cls_logits = p[..., 5:]
        box = p[..., :4].sigmoid()
        obj_target = torch.zeros((bsz, h, w), device=device)
        cls_target = torch.zeros((bsz, h, w, nc), device=device)
        box_target = torch.zeros((bsz, h, w, 4), device=device)
        pos_mask = torch.zeros((bsz, h, w), dtype=torch.bool, device=device)
        for bi, t in enumerate(targets):
            if t.numel() == 0:
                continue
            gt = t.to(device)
            cls_idx = gt[:, 0].long().clamp(0, nc - 1)
            cx = gt[:, 1].clamp(0, 1 - 1e-6)
            cy = gt[:, 2].clamp(0, 1 - 1e-6)
            bw = gt[:, 3].clamp(0, 1)
            bh = gt[:, 4].clamp(0, 1)

            cx_grid = cx * w
            cy_grid = cy * h
            gx = cx_grid.long().clamp(0, w - 1)
            gy = cy_grid.long().clamp(0, h - 1)
            tx = cx_grid - gx.float()
            ty = cy_grid - gy.float()

            obj_target[bi, gy, gx] = 1.0
            cls_target[bi, gy, gx, cls_idx] = 1.0
            # Simplified YOLO-style assignment: if multiple GT boxes land in the
            # same cell, later writes overwrite earlier ones without error. The
            # decode inverse is cx_abs ~= (gx + tx) / feature_w and
            # cy_abs ~= (gy + ty) / feature_h; widths/heights remain full-image
            # normalized values.
            box_target[bi, gy, gx, 0] = tx
            box_target[bi, gy, gx, 1] = ty
            box_target[bi, gy, gx, 2] = bw
            box_target[bi, gy, gx, 3] = bh
            pos_mask[bi, gy, gx] = True
        obj_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(obj, obj_target))
        if pos_mask.any():
            cls_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(cls_logits[pos_mask], cls_target[pos_mask]))
            box_terms.append(torch.nn.functional.l1_loss(box[pos_mask], box_target[pos_mask]))
        else:
            cls_terms.append(cls_logits.sum() * 0.0)
            box_terms.append(box.sum() * 0.0)
    loss_box = torch.stack(box_terms).mean()
    loss_obj = torch.stack(obj_terms).mean()
    loss_cls = torch.stack(cls_terms).mean()
    return {"loss_box": loss_box, "loss_obj": loss_obj, "loss_cls": loss_cls, "loss_total": loss_box + loss_obj + loss_cls}


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return box1.new_zeros((box1.shape[0], box2.shape[0]))
    a = torch.max(box1[:, None, :2], box2[None, :, :2])
    b = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    inter = (b - a).clamp(min=0).prod(2)
    area1 = (box1[:, 2:] - box1[:, :2]).clamp(min=0).prod(1)
    area2 = (box2[:, 2:] - box2[:, :2]).clamp(min=0).prod(1)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(1e-6)
