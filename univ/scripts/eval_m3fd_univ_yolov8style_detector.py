"""Evaluate UNIV-M3FD RGB-IR YOLOv8-style detector."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from univ.scripts.datasets.m3fd_rgbt_detection_dataset import M3FDRGBTDetectionDataset
from univ.scripts.m3fd_train_eval_common import CLASS_NAMES, NC, box_iou, collate_m3fd, load_yaml, xywhn_to_xyxy_abs
from univ.scripts.models.m3fd_univ_rgbt_yolov8style_detector import M3FDUNIVRGBTYOLOv8StyleDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="univ/configs/m3fd.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=None)
    return p.parse_args()


def decode(preds, imgsz: int, conf_thres: float):
    out = []
    for p in preds:
        b, h, w, _ = p.shape
        obj = p[..., 4].sigmoid()
        cls_prob = p[..., 5:].sigmoid()
        score, cls = (obj[..., None] * cls_prob).max(-1)
        box = p[..., :4].sigmoid()
        yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing="ij")
        cx = (xx[None] + box[..., 0]) / w * imgsz
        cy = (yy[None] + box[..., 1]) / h * imgsz
        bw = box[..., 2] * imgsz
        bh = box[..., 3] * imgsz
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        for bi in range(b):
            mask = score[bi] > conf_thres
            out.append(torch.cat([xyxy[bi][mask], score[bi][mask, None], cls[bi][mask, None].float()], 1) if mask.any() else p.new_zeros((0, 6)))
    # merge levels by image index
    merged = []
    bs = preds[0].shape[0]
    for bi in range(bs):
        parts = [out[level * bs + bi] for level in range(len(preds))]
        merged.append(torch.cat(parts, 0) if parts else preds[0].new_zeros((0, 6)))
    return merged


def _average_precision(tp_flags: torch.Tensor, scores: torch.Tensor, n_gt: int) -> torch.Tensor:
    if n_gt == 0:
        return scores.new_tensor(float("nan"))
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    order = torch.argsort(scores, descending=True)
    tp_sorted = tp_flags[order].float()
    fp_sorted = 1.0 - tp_sorted
    tp_cum = torch.cumsum(tp_sorted, 0)
    fp_cum = torch.cumsum(fp_sorted, 0)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / (tp_cum + fp_cum).clamp_min(1e-9)
    mrec = torch.cat([recall.new_tensor([0.0]), recall, recall.new_tensor([1.0])])
    mpre = torch.cat([precision.new_tensor([1.0]), precision, precision.new_tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    return torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def class_aware_ap(detections, targets, iou_thresholds: torch.Tensor, nc: int):
    per_thr_cls = torch.zeros((len(iou_thresholds), nc), dtype=torch.float32)
    gt_counts = torch.zeros(nc, dtype=torch.float32)
    for ti, threshold in enumerate(iou_thresholds):
        for c in range(nc):
            scores = []
            tp_flags = []
            n_gt = 0
            for det, gt in zip(detections, targets):
                pred_c = det[det[:, 5].long() == c] if det.numel() else det.new_zeros((0, 6))
                gt_c = gt[gt[:, 0].long() == c] if gt.numel() else gt.new_zeros((0, 5))
                n_gt += gt_c.shape[0]
                if ti == 0:
                    gt_counts[c] += gt_c.shape[0]
                if pred_c.numel() == 0:
                    continue
                order = torch.argsort(pred_c[:, 4], descending=True)
                pred_c = pred_c[order]
                matched_gt = set()
                for pred in pred_c:
                    scores.append(pred[4].detach().cpu())
                    if gt_c.numel() == 0:
                        tp_flags.append(torch.tensor(0.0))
                        continue
                    ious = box_iou(pred[None, :4], gt_c[:, 1:5]).squeeze(0)
                    best_iou, best_gt = ious.max(0)
                    gi = int(best_gt)
                    if best_iou >= threshold and gi not in matched_gt:
                        tp_flags.append(torch.tensor(1.0))
                        matched_gt.add(gi)
                    else:
                        tp_flags.append(torch.tensor(0.0))
            score_tensor = torch.stack(scores) if scores else torch.zeros(0)
            tp_tensor = torch.stack(tp_flags) if tp_flags else torch.zeros(0)
            per_thr_cls[ti, c] = _average_precision(tp_tensor, score_tensor, n_gt)
    valid = gt_counts > 0
    return per_thr_cls, valid


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    data = args.data or cfg.get("data") or cfg.get("dataset", {}).get("path")
    imgsz = args.imgsz or int(cfg.get("imgsz", 224))
    batch = args.batch or int(cfg.get("batch", 4))
    fusion = cfg.get("fusion", "concat")
    device_arg = args.device
    if device_arg and device_arg.isdigit():
        device_arg = f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = M3FDUNIVRGBTYOLOv8StyleDetector(nc=NC, imgsz=imgsz, fusion=fusion).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    ds = M3FDRGBTDetectionDataset(data, split=args.split, imgsz=imgsz)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_m3fd)
    iou_thresholds = torch.arange(0.50, 0.96, 0.05)
    all_dets = []
    all_targets = []
    with torch.no_grad():
        for b in dl:
            preds = model(b["visible"].to(device), b["infrared"].to(device))
            dets = decode(preds, imgsz, args.conf)
            for det, target in zip(dets, b["targets"]):
                gt = xywhn_to_xyxy_abs(target, imgsz, imgsz).to(det.device)
                all_dets.append(det.detach().cpu())
                all_targets.append(gt.detach().cpu())
    per_thr_cls, valid_classes = class_aware_ap(all_dets, all_targets, iou_thresholds, NC)
    ap50 = per_thr_cls[0]
    ap5095 = torch.nanmean(per_thr_cls, 0)
    valid_ap50 = ap50[valid_classes]
    valid_ap5095 = per_thr_cls[:, valid_classes]
    # Dataset-level precision/recall use the same class-aware matching at IoU=0.50.
    tp = torch.zeros(NC); fp = torch.zeros(NC); fn = torch.zeros(NC)
    for det, gt in zip(all_dets, all_targets):
        for c in range(NC):
            pred_c = det[det[:, 5].long() == c] if det.numel() else det.new_zeros((0, 6))
            gt_c = gt[gt[:, 0].long() == c] if gt.numel() else gt.new_zeros((0, 5))
            matched_gt = set()
            if pred_c.numel():
                pred_c = pred_c[torch.argsort(pred_c[:, 4], descending=True)]
            for pred in pred_c:
                if gt_c.numel() == 0:
                    fp[c] += 1
                    continue
                ious = box_iou(pred[None, :4], gt_c[:, 1:5]).squeeze(0)
                best_iou, best_gt = ious.max(0)
                gi = int(best_gt)
                if best_iou >= 0.5 and gi not in matched_gt:
                    tp[c] += 1
                    matched_gt.add(gi)
                else:
                    fp[c] += 1
            fn[c] += max(0, gt_c.shape[0] - len(matched_gt))
    precision = (tp.sum() / (tp.sum() + fp.sum()).clamp_min(1)).item()
    recall = (tp.sum() / (tp.sum() + fn.sum()).clamp_min(1)).item()
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"mAP50: {torch.nanmean(valid_ap50).item() if valid_ap50.numel() else 0.0:.6f}")
    print(f"mAP50:95: {torch.nanmean(valid_ap5095).item() if valid_ap5095.numel() else 0.0:.6f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"AP50/{name}: {ap50[i].item():.6f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"AP50:95/{name}: {ap5095[i].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
