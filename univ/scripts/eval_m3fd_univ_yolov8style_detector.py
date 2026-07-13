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


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    data = args.data or cfg.get("data") or cfg.get("dataset", {}).get("path")
    imgsz = args.imgsz or int(cfg.get("imgsz", 224))
    batch = args.batch or int(cfg.get("batch", 4))
    fusion = cfg.get("fusion", "concat")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = M3FDUNIVRGBTYOLOv8StyleDetector(nc=NC, imgsz=imgsz, fusion=fusion).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    ds = M3FDRGBTDetectionDataset(data, split=args.split, imgsz=imgsz)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_m3fd)
    tp = torch.zeros(NC); fp = torch.zeros(NC); fn = torch.zeros(NC)
    ap50 = torch.zeros(NC)
    iou_thresholds = torch.arange(0.50, 0.96, 0.05)
    map_hits = []
    with torch.no_grad():
        for b in dl:
            preds = model(b["visible"].to(device), b["infrared"].to(device))
            dets = decode(preds, imgsz, args.conf)
            for det, target in zip(dets, b["targets"]):
                gt = xywhn_to_xyxy_abs(target, imgsz, imgsz).to(det.device)
                matched = set()
                for c in range(NC):
                    d = det[det[:, 5].long() == c]
                    g = gt[gt[:, 0].long() == c]
                    if d.numel() == 0:
                        fn[c] += len(g); continue
                    if g.numel() == 0:
                        fp[c] += len(d); continue
                    ious = box_iou(d[:, :4], g[:, 1:5])
                    best_iou, best_gt = ious.max(1)
                    for i, iou in enumerate(best_iou):
                        gi = int(best_gt[i])
                        if iou >= 0.5 and gi not in matched:
                            tp[c] += 1; matched.add(gi)
                        else:
                            fp[c] += 1
                    fn[c] += max(0, len(g) - len(matched))
                    ap50[c] = tp[c] / (tp[c] + fp[c] + fn[c]).clamp_min(1)
                if gt.numel():
                    all_iou = box_iou(det[:, :4], gt[:, 1:5]) if det.numel() else gt.new_zeros((0, gt.shape[0]))
                    map_hits.extend([(all_iou.max(0).values >= t).float().mean().item() if all_iou.numel() else 0.0 for t in iou_thresholds])
    precision = (tp.sum() / (tp.sum() + fp.sum()).clamp_min(1)).item()
    recall = (tp.sum() / (tp.sum() + fn.sum()).clamp_min(1)).item()
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"mAP50: {ap50.mean().item():.6f}")
    print(f"mAP50:95: {(sum(map_hits) / max(1, len(map_hits))):.6f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"AP50/{name}: {ap50[i].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
