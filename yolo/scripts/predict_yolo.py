#!/usr/bin/env python3
from __future__ import annotations
import argparse
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights', required=True); ap.add_argument('--source', required=True); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='0'); ap.add_argument('--conf', type=float, default=0.25); ap.add_argument('--project', default='yolo/runs/predict'); ap.add_argument('--name', default='predict'); ap.add_argument('--modality', choices=['visible','infrared'], default='visible')
    a=ap.parse_args(); print(f'Predicting {a.modality}: {a}', flush=True)
    YOLO(a.weights).predict(source=a.source, imgsz=a.imgsz, device=a.device, conf=a.conf, project=a.project, name=a.name, save=True)
if __name__=='__main__': main()
