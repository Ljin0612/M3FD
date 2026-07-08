# M3FD Official YOLO Baselines

This folder is an independent YOLO baseline implementation for M3FD detection. It does not modify or import the original `UNIV-main/` code.

## Baselines

1. **visible-only**: train/validate/test official Ultralytics YOLO on `vi` images only.
2. **infrared-only**: train/validate/test official Ultralytics YOLO on `ir` images only.
3. **late-fusion multimodal**: load the visible YOLO `best.pt` and infrared YOLO `best.pt`, run both at test time, merge boxes, and compute fused detection metrics.

The first late-fusion implementation uses **class-wise NMS** at box level. It is a test-stage fusion baseline, **not** an end-to-end two-stream YOLO model.

Classes are fixed:

| id | class |
|---:|---|
| 0 | people |
| 1 | car |
| 2 | bus |
| 3 | motorcycle |
| 4 | lamp |
| 5 | truck |

## 1. Environment installation

```bash
cd /home/jinlei/code/M3FD
pip install -r yolo/requirements.txt
```

## 2. Data preparation

Default raw dataset path:

```text
/home/jinlei/database/M3FD_Detection
```

Expected raw structure:

```text
M3FD_Detection/
├── vi/
├── ir/
├── labels/
└── meta/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Prepare a YOLO-format dataset with symlinks:

```bash
python yolo/scripts/prepare_m3fd_yolo.py \
  --root /home/jinlei/database/M3FD_Detection \
  --out yolo/datasets/m3fd_yolo \
  --mode symlink
```

If symlinks are not supported, use:

```bash
python yolo/scripts/prepare_m3fd_yolo.py \
  --root /home/jinlei/database/M3FD_Detection \
  --out yolo/datasets/m3fd_yolo \
  --mode copy
```

The preparation script creates:

```text
yolo/datasets/m3fd_yolo/visible/images/{train,val,test}
yolo/datasets/m3fd_yolo/visible/labels/{train,val,test}
yolo/datasets/m3fd_yolo/infrared/images/{train,val,test}
yolo/datasets/m3fd_yolo/infrared/labels/{train,val,test}
```

It also writes `yolo/configs/m3fd_visible.yaml` and `yolo/configs/m3fd_infrared.yaml`. Labels are checked as YOLO `class cx cy w h` with normalized coordinates in `[0, 1]`; abnormal labels are printed as warnings. The preparation script is safe to run repeatedly: before repopulating each modality/split, it refreshes the managed `images/<split>` and `labels/<split>` output directories so stale symlinks or copied files from older meta splits are removed.

## 3. Data checking

```bash
python yolo/scripts/check_m3fd_yolo.py \
  --dataset yolo/datasets/m3fd_yolo
```

The checker reports train/val/test image counts, label counts, missing images, missing labels, empty labels, and per-class object counts.

## 4. visible-only training

```bash
python yolo/scripts/train_yolo.py \
  --data yolo/configs/m3fd_visible.yaml \
  --model yolov8s.pt \
  --epochs 300 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project yolo/runs/detect \
  --name yolov8s_visible_e300_b8
```

## 5. infrared-only training

```bash
python yolo/scripts/train_yolo.py \
  --data yolo/configs/m3fd_infrared.yaml \
  --model yolov8s.pt \
  --epochs 300 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project yolo/runs/detect \
  --name yolov8s_infrared_e300_b8
```

## 6. visible-only test

```bash
python yolo/scripts/eval_yolo.py \
  --weights yolo/runs/detect/yolov8s_visible_e300_b8/weights/best.pt \
  --data yolo/configs/m3fd_visible.yaml \
  --split test \
  --imgsz 640 \
  --device 0 \
  --name yolov8s_visible_test
```

Outputs:

```text
yolo/results/yolov8s_visible_test_eval_summary.md
yolo/results/yolov8s_visible_test_eval_summary.csv
```

## 7. infrared-only test

```bash
python yolo/scripts/eval_yolo.py \
  --weights yolo/runs/detect/yolov8s_infrared_e300_b8/weights/best.pt \
  --data yolo/configs/m3fd_infrared.yaml \
  --split test \
  --imgsz 640 \
  --device 0 \
  --name yolov8s_infrared_test
```

## 8. Prediction visualization

```bash
python yolo/scripts/predict_yolo.py \
  --weights yolo/runs/detect/yolov8s_visible_e300_b8/weights/best.pt \
  --source yolo/datasets/m3fd_yolo/visible/images/test \
  --modality visible \
  --project yolo/runs/predict \
  --name visible_test_vis
```

Use `--modality infrared` and an infrared source folder for infrared visualization.

## 9. late-fusion test

```bash
python yolo/scripts/late_fusion_eval.py \
  --visible-weights yolo/runs/detect/yolov8s_visible_e300_b8/weights/best.pt \
  --infrared-weights yolo/runs/detect/yolov8s_infrared_e300_b8/weights/best.pt \
  --root /home/jinlei/database/M3FD_Detection \
  --split test \
  --imgsz 640 \
  --device 0 \
  --conf 0.001 \
  --iou 0.6 \
  --fusion-iou 0.55 \
  --project yolo/runs/fusion \
  --name yolov8s_late_fusion_test
```

Outputs:

```text
yolo/runs/fusion/yolov8s_late_fusion_test/fusion_eval_summary.md
yolo/runs/fusion/yolov8s_late_fusion_test/fusion_eval_summary.csv
yolo/runs/fusion/yolov8s_late_fusion_test/predictions.json
```

The metrics implementation in `yolo/tools/metrics.py` is standalone and does not depend on `UNIV-main/`.

## 10. Result collection

```bash
python yolo/scripts/collect_yolo_results.py \
  --results-dir yolo/results \
  --fusion-dir yolo/runs/fusion
```

Outputs:

```text
yolo/results/yolo_baseline_table.csv
yolo/results/yolo_baseline_table.md
```

## Notes and future extensions

This baseline intentionally prioritizes stable, official YOLO training and evaluation. The late-fusion baseline is a test-stage, box-level fusion method using class-wise NMS. If needed later, this folder can be extended with 6-channel early fusion or a two-stream YOLO architecture, but those are outside this first stable baseline version.
