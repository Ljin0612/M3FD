#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from tools.metrics import BoxRecord, CLASS_NAMES, ap_per_class, class_wise_nms, load_yolo_labels
IMG_EXTS=['.jpg','.jpeg','.png','.bmp','.tif','.tiff']
def p(msg): print(msg, flush=True)
def ids(root, split): return [Path(x.strip()).stem for x in (root/'meta'/f'{split}.txt').read_text().splitlines() if x.strip()]
def find_img(folder, stem):
    for e in IMG_EXTS:
        for q in (folder/(stem+e), folder/(stem+e.upper())):
            if q.exists(): return q
    m=list(folder.glob(stem+'.*')); return m[0] if m else None
def preds(model, img, image_id, imgsz, device, conf, iou):
    rs=model.predict(source=str(img), imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
    out=[]
    for r in rs:
        if r.boxes is None: continue
        xyxy=r.boxes.xyxyn.cpu().numpy(); cls=r.boxes.cls.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
        for b,c,s in zip(xyxy,cls,cf): out.append(BoxRecord(image_id, int(c), tuple(map(float,b)), float(s)))
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--visible-weights', required=True); ap.add_argument('--infrared-weights', required=True); ap.add_argument('--root', default='/home/jinlei/database/M3FD_Detection'); ap.add_argument('--split', default='test'); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='0'); ap.add_argument('--conf', type=float, default=0.001); ap.add_argument('--iou', type=float, default=0.6); ap.add_argument('--fusion-iou', type=float, default=0.55); ap.add_argument('--project', default='yolo/runs/fusion'); ap.add_argument('--name', default='yolov8s_late_fusion_test')
    a=ap.parse_args(); root=Path(a.root); out=Path(a.project)/a.name; out.mkdir(parents=True, exist_ok=True)
    p(f'Late fusion eval (class-wise NMS): {a}')
    vm, im=YOLO(a.visible_weights), YOLO(a.infrared_weights)
    all_preds=[]; all_gts=[]
    for stem in ids(root, a.split):
        vi=find_img(root/'vi', stem); ir=find_img(root/'ir', stem)
        if vi is None or ir is None:
            p(f'WARNING: missing modality image for {stem}'); continue
        merged=preds(vm, vi, stem, a.imgsz, a.device, a.conf, a.iou)+preds(im, ir, stem, a.imgsz, a.device, a.conf, a.iou)
        fused=class_wise_nms(merged, a.fusion_iou); all_preds.extend(fused)
        all_gts.extend(load_yolo_labels(root/'labels'/f'{stem}.txt', stem))
    metrics=ap_per_class(all_preds, all_gts)
    rows={'Precision':metrics['precision'],'Recall':metrics['recall'],'mAP50':metrics['map50'],'mAP50:95':metrics['map5095']}
    for n in CLASS_NAMES: rows[f'AP50/{n}']=metrics['ap50_per_class'][n]; rows[f'AP50:95/{n}']=metrics['ap5095_per_class'][n]
    with (out/'fusion_eval_summary.csv').open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=list(rows)); w.writeheader(); w.writerow(rows)
    (out/'fusion_eval_summary.md').write_text('|Metric|Value|\n|---|---|\n'+''.join(f'|{k}|{v}|\n' for k,v in rows.items()))
    (out/'predictions.json').write_text(json.dumps([r.__dict__ for r in all_preds], indent=2))
    p(f'Wrote outputs to {out}')
if __name__=='__main__': main()
