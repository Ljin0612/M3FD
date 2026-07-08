#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
NAMES=['people','car','bus','motorcycle','lamp','truck']; IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
def p(msg): print(msg, flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset', default='yolo/datasets/m3fd_yolo'); args=ap.parse_args(); root=Path(args.dataset)
    total=Counter()
    for mod in ['visible','infrared']:
      p(f'[{mod}]')
      for split in ['train','val','test']:
        img_dir=root/mod/'images'/split; lab_dir=root/mod/'labels'/split
        imgs=[x for x in img_dir.iterdir() if x.suffix.lower() in IMG_EXTS] if img_dir.exists() else []
        labs=list(lab_dir.glob('*.txt')) if lab_dir.exists() else []
        img_stems={x.stem for x in imgs}; lab_stems={x.stem for x in labs}
        missing_labels=sorted(img_stems-lab_stems); missing_images=sorted(lab_stems-img_stems)
        empty=[x.name for x in labs if x.stat().st_size==0]
        cnt=Counter()
        for lab in labs:
          for line in lab.read_text().splitlines():
            parts=line.split()
            if parts:
              try: cnt[int(float(parts[0]))]+=1
              except ValueError: pass
        total.update(cnt)
        p(f'  {split}: images={len(imgs)} labels={len(labs)} missing_labels={len(missing_labels)} missing_images={len(missing_images)} empty_labels={len(empty)}')
        if missing_labels[:10]: p(f'    missing label examples: {missing_labels[:10]}')
        if missing_images[:10]: p(f'    missing image examples: {missing_images[:10]}')
        p('    class_counts: '+', '.join(f'{i}/{NAMES[i]}={cnt[i]}' for i in range(len(NAMES))))
    p('[summary] total class counts across modalities/splits: '+', '.join(f'{i}/{NAMES[i]}={total[i]}' for i in range(len(NAMES))))
if __name__ == '__main__': main()
