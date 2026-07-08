#!/usr/bin/env python3
from __future__ import annotations
import argparse
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--model', default='yolov8s.pt'); ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='0'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--project', default='yolo/runs/detect'); ap.add_argument('--name', default='yolov8s_m3fd'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True); ap.add_argument('--resume', action='store_true')
    a=ap.parse_args(); print(f'Training YOLO: {a}', flush=True)
    YOLO(a.model).train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device=a.device, workers=a.workers, project=a.project, name=a.name, seed=a.seed, pretrained=a.pretrained, resume=a.resume)
if __name__=='__main__': main()
