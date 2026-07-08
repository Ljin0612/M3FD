#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
FIELDS=['Model','Input','Fusion','Epochs','Batch','ImgSz','Precision','Recall','mAP50','mAP50:95']
def read_first_csv(p):
    with p.open() as f: return next(csv.DictReader(f))
def infer(name):
    low=name.lower(); return ('visible' if 'visible' in low else 'infrared' if 'infrared' in low else 'multi-modal', 'late' if 'fusion' in low else 'none')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir', default='yolo/results'); ap.add_argument('--fusion-dir', default='yolo/runs/fusion'); args=ap.parse_args(); rows=[]
    for f in Path(args.results_dir).glob('*_eval_summary.csv'):
        r=read_first_csv(f); inp, fus=infer(f.stem); rows.append({'Model':r.get('Model',f.stem.replace('_eval_summary','')),'Input':inp,'Fusion':fus,'Epochs':'','Batch':'','ImgSz':'','Precision':r.get('Precision',''),'Recall':r.get('Recall',''),'mAP50':r.get('mAP50',''),'mAP50:95':r.get('mAP50:95','')})
    for f in Path(args.fusion_dir).glob('*/fusion_eval_summary.csv'):
        r=read_first_csv(f); rows.append({'Model':f.parent.name,'Input':'visible+infrared','Fusion':'class-wise NMS','Epochs':'','Batch':'','ImgSz':'','Precision':r.get('Precision',''),'Recall':r.get('Recall',''),'mAP50':r.get('mAP50',''),'mAP50:95':r.get('mAP50:95','')})
    out=Path(args.results_dir); out.mkdir(parents=True, exist_ok=True); csvp=out/'yolo_baseline_table.csv'; mdp=out/'yolo_baseline_table.md'
    with csvp.open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    md='|'+'|'.join(FIELDS)+'|\n|'+'|'.join(['---']*len(FIELDS))+'|\n' + ''.join('|'+'|'.join(str(r.get(k,'')) for k in FIELDS)+'|\n' for r in rows)
    mdp.write_text(md); print(f'Wrote {csvp} and {mdp}', flush=True)
if __name__=='__main__': main()
