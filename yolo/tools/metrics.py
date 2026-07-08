#!/usr/bin/env python3
"""Standalone detection metrics for YOLO-format M3FD labels/predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

CLASS_NAMES = ["people", "car", "bus", "motorcycle", "lamp", "truck"]
NC = len(CLASS_NAMES)

@dataclass
class BoxRecord:
    image_id: str
    cls: int
    box: Tuple[float, float, float, float]  # xyxy normalized or absolute, consistently
    conf: float = 1.0


def xywhn_to_xyxy(vals: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = map(float, vals)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def xyxy_iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=float)
    b = np.asarray(box, dtype=float)
    inter_x1 = np.maximum(b[0], boxes[:, 0])
    inter_y1 = np.maximum(b[1], boxes[:, 1])
    inter_x2 = np.minimum(b[2], boxes[:, 2])
    inter_y2 = np.minimum(b[3], boxes[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area1 = np.maximum(0.0, b[2] - b[0]) * np.maximum(0.0, b[3] - b[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-12)


def class_wise_nms(preds: Iterable[BoxRecord], iou_thr: float = 0.55) -> List[BoxRecord]:
    output: List[BoxRecord] = []
    groups: Dict[Tuple[str, int], List[BoxRecord]] = {}
    for p in preds:
        groups.setdefault((p.image_id, p.cls), []).append(p)
    for _, items in groups.items():
        items = sorted(items, key=lambda x: x.conf, reverse=True)
        kept: List[BoxRecord] = []
        while items:
            best = items.pop(0)
            kept.append(best)
            if not items:
                break
            boxes = np.asarray([x.box for x in items], dtype=float)
            ious = xyxy_iou(best.box, boxes)
            items = [x for x, iou in zip(items, ious) if iou <= iou_thr]
        output.extend(kept)
    return output


def load_yolo_labels(label_path: Path, image_id: str) -> List[BoxRecord]:
    records: List[BoxRecord] = []
    if not label_path.exists():
        return records
    for line_no, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label {label_path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        records.append(BoxRecord(image_id=image_id, cls=cls, box=xywhn_to_xyxy([float(x) for x in parts[1:5]]), conf=1.0))
    return records


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(preds: List[BoxRecord], gts: List[BoxRecord], iou_thresholds: Sequence[float] | None = None, nc: int = NC) -> Dict:
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)
    iou_thresholds = np.asarray(iou_thresholds, dtype=float)
    ap = np.zeros((nc, len(iou_thresholds)), dtype=float)
    precision50 = np.zeros(nc, dtype=float)
    recall50 = np.zeros(nc, dtype=float)
    gt_by_key: Dict[Tuple[str, int], List[BoxRecord]] = {}
    for gt in gts:
        if 0 <= gt.cls < nc:
            gt_by_key.setdefault((gt.image_id, gt.cls), []).append(gt)
    for cls in range(nc):
        cls_preds = sorted([p for p in preds if p.cls == cls], key=lambda x: x.conf, reverse=True)
        n_gt = sum(len(v) for (img, c), v in gt_by_key.items() if c == cls)
        if n_gt == 0:
            continue
        for t_idx, thr in enumerate(iou_thresholds):
            matched: Dict[Tuple[str, int], bool] = {}
            tp = np.zeros(len(cls_preds), dtype=float)
            fp = np.zeros(len(cls_preds), dtype=float)
            for i, pred in enumerate(cls_preds):
                candidates = gt_by_key.get((pred.image_id, cls), [])
                boxes = np.asarray([g.box for g in candidates], dtype=float)
                ious = xyxy_iou(pred.box, boxes)
                if ious.size:
                    candidate_idxs = np.flatnonzero(ious >= thr)
                    unmatched_idxs = [
                        int(j) for j in candidate_idxs
                        if not matched.get((pred.image_id, int(j)), False)
                    ]
                    if unmatched_idxs:
                        j = max(unmatched_idxs, key=lambda idx: float(ious[idx]))
                        tp[i] = 1.0
                        matched[(pred.image_id, j)] = True
                    else:
                        fp[i] = 1.0
                else:
                    fp[i] = 1.0
            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap[cls, t_idx] = compute_ap(recall, precision) if len(cls_preds) else 0.0
            if abs(thr - 0.5) < 1e-9:
                precision50[cls] = float(precision[-1]) if len(precision) else 0.0
                recall50[cls] = float(recall[-1]) if len(recall) else 0.0
    present = np.array([sum(len(v) for (img, c), v in gt_by_key.items() if c == cls) > 0 for cls in range(nc)])
    return {
        "precision": float(precision50[present].mean()) if present.any() else 0.0,
        "recall": float(recall50[present].mean()) if present.any() else 0.0,
        "map50": float(ap[present, 0].mean()) if present.any() else 0.0,
        "map5095": float(ap[present, :].mean()) if present.any() else 0.0,
        "ap50_per_class": {CLASS_NAMES[i]: float(ap[i, 0]) for i in range(nc)},
        "ap5095_per_class": {CLASS_NAMES[i]: float(ap[i, :].mean()) for i in range(nc)},
    }
