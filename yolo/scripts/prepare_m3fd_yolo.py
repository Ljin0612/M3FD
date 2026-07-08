#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

NAMES = {0: 'people', 1: 'car', 2: 'bus', 3: 'motorcycle', 4: 'lamp', 5: 'truck'}
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def p(msg): print(msg, flush=True)

def read_ids(meta_file: Path):
    ids=[]
    for line in meta_file.read_text().splitlines():
        s=line.strip()
        if not s: continue
        ids.append(Path(s).stem)
    return ids

def find_image(folder: Path, stem: str):
    for ext in IMG_EXTS:
        q=folder/(stem+ext)
        if q.exists(): return q
        q=folder/(stem+ext.upper())
        if q.exists(): return q
    matches=list(folder.glob(stem+'.*'))
    return matches[0] if matches else None

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    if mode == 'copy': shutil.copy2(src, dst)
    else: dst.symlink_to(src.resolve())

def check_label(path: Path):
    bad=False
    if not path.exists(): return False
    for i,line in enumerate(path.read_text().splitlines(),1):
        parts=line.split()
        if not parts: continue
        if len(parts)!=5:
            p(f'WARNING: label format should be 5 columns: {path}:{i}') ; bad=True; continue
        try:
            cls=int(float(parts[0])); vals=[float(x) for x in parts[1:]]
        except ValueError:
            p(f'WARNING: non-numeric label: {path}:{i}') ; bad=True; continue
        if cls not in NAMES or any(v < 0 or v > 1 for v in vals):
            p(f'WARNING: invalid YOLO values in {path}:{i}: {line}') ; bad=True
    return not bad

def write_yaml(path: Path, dataset_root: Path, modality: str):
    text = (
        f"path: {str((dataset_root / modality).resolve())}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + ''.join(f"  {i}: {name}\n" for i, name in NAMES.items())
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    p(f'Wrote {path}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/jinlei/database/M3FD_Detection')
    ap.add_argument('--out', default='yolo/datasets/m3fd_yolo')
    ap.add_argument('--mode', choices=['symlink','copy'], default='symlink')
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.out)
    for d in [root/'vi', root/'ir', root/'labels', root/'meta']:
        if not d.exists(): raise FileNotFoundError(d)
    counts={}
    for split in ['train','val','test']:
        meta=root/'meta'/f'{split}.txt'
        if not meta.exists(): raise FileNotFoundError(meta)
        ids=read_ids(meta); counts[split]=len(ids)
        for stem in ids:
            lab=root/'labels'/f'{stem}.txt'
            if lab.exists(): check_label(lab)
            else: p(f'WARNING: missing label: {lab}')
            for modality, folder in [('visible', root/'vi'), ('infrared', root/'ir')]:
                img=find_image(folder, stem)
                if img is None:
                    p(f'WARNING: missing {modality} image for {stem} in {folder}')
                    continue
                link_or_copy(img, out/modality/'images'/split/img.name, args.mode)
                if lab.exists(): link_or_copy(lab, out/modality/'labels'/split/lab.name, args.mode)
    write_yaml(Path('yolo/configs/m3fd_visible.yaml'), out, 'visible')
    write_yaml(Path('yolo/configs/m3fd_infrared.yaml'), out, 'infrared')
    p(f'Done. splits={counts}')
if __name__ == '__main__': main()
