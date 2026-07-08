#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
from ultralytics import YOLO
NAMES=['people','car','bus','motorcycle','lamp','truck']
def scalar(x):
    try: return float(x)
    except Exception: return 0.0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights', required=True); ap.add_argument('--data', required=True); ap.add_argument('--split', choices=['val','test'], default='test'); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='0'); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--project', default='yolo/runs/detect'); ap.add_argument('--name', required=True); ap.add_argument('--results-dir', default='yolo/results')
    a=ap.parse_args(); print(f'Evaluating YOLO: {a}', flush=True)
    m=YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device, batch=a.batch, project=a.project, name=a.name)
    box=m.box; maps=getattr(box,'maps',[]) or []
    row={'Model':a.name,'Precision':scalar(box.mp),'Recall':scalar(box.mr),'mAP50':scalar(box.map50),'mAP50:95':scalar(box.map)}
    for i,n in enumerate(NAMES): row[f'AP50:95/{n}']=scalar(maps[i]) if i < len(maps) else 0.0
    out=Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    csvp=out/f'{a.name}_eval_summary.csv'; mdp=out/f'{a.name}_eval_summary.md'
    with csvp.open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    mdp.write_text('|Metric|Value|\n|---|---|\n' + ''.join(f'|{k}|{v}|\n' for k,v in row.items()))
    print(f'Wrote {csvp} and {mdp}', flush=True)
if __name__=='__main__': main()
