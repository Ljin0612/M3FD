# UNIV-M3FD Experiment Guide

## 1. UNIV 文件夹说明

- 原始 UNIV 代码已整理到 `univ/`。仓库中原上传目录名为 `UNIV-main/`，本次改为 `univ/` 以匹配 M3FD 仓库约定。
- `univ/` 当前定位是跨模态统一表征 baseline，包含原始 MCMAE/UNIV 预训练代码、backbone、loss、dataset、segmentation adapter 代码和轻量环境/数据检查脚本。
- `yolo/` 是官方 YOLO 强检测 baseline；`ours/` 是后续改进方法。本次未修改 `yolo/` 或 `ours/`。
- 当前仓库未发现已完成的 `train_m3fd_univ_yolov8style_detector.py` 检测训练入口；因此本文档区分“当前可直接运行的原始 UNIV 脚本”和“后续需要适配 M3FD 检测的脚本”。

## 2. 环境安装流程

```bash
conda create -n univ_m3fd python=3.10 -y
conda activate univ_m3fd
```

PyTorch 请按服务器 CUDA 版本单独安装，例如到 <https://pytorch.org/get-started/locally/> 选择匹配命令。不要从 `univ/requirements.txt` 安装固定 CUDA wheel。

```bash
pip install -r univ/requirements.txt
python univ/scripts/check_univ_env.py
```

复杂/可选依赖：`univ/SEG/MCMAE_SEG/` 是原始 mmseg/mmcv 语义分割适配代码，可能需要 `mmcv`、`mmsegmentation`、`apex` 等旧 OpenMMLab 依赖。它们不是当前 M3FD RGB-IR 检测主流程的必需依赖，因此没有写入 `univ/requirements.txt`。

## 3. 数据集准备

默认 M3FD 路径：

```text
/home/jinlei/database/M3FD_Detection
```

期望结构：

```text
M3FD_Detection/
├── vi/
├── ir/
├── labels/
├── Annotation/
└── meta/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

类别：

```text
0 people
1 car
2 bus
3 motorcycle
4 lamp
5 truck
```

## 4. 数据检查命令

```bash
python univ/scripts/check_m3fd_for_univ.py --root /home/jinlei/database/M3FD_Detection
find /home/jinlei/database/M3FD_Detection/vi -maxdepth 1 -type f | wc -l
find /home/jinlei/database/M3FD_Detection/ir -maxdepth 1 -type f | wc -l
find /home/jinlei/database/M3FD_Detection/labels -maxdepth 1 -type f | wc -l
wc -l /home/jinlei/database/M3FD_Detection/meta/train.txt /home/jinlei/database/M3FD_Detection/meta/val.txt /home/jinlei/database/M3FD_Detection/meta/test.txt
```

## 5. UNIV 原始权重准备

建议放置路径：

```text
univ/pretrained/checkpoint0400.pth
```

- no-pretrained 实验不需要该权重。
- pretrained 实验需要通过 `--univ-weights univ/pretrained/checkpoint0400.pth` 指定。
- 本次加入了 `torch_load` 兼容封装，统一设置 `map_location`，并兼容 PyTorch 2.4+ 的 `weights_only` 参数。

## 6. Smoke test 流程

### 当前可直接运行的原始 UNIV 脚本

环境检查：

```bash
python univ/scripts/check_univ_env.py
python univ/scripts/check_m3fd_for_univ.py --root /home/jinlei/database/M3FD_Detection
python -m py_compile $(find univ -name "*.py")
```

原始 UNIV 预训练入口：

```bash
PYTHONUNBUFFERED=1 python -u univ/pretrain_mcmae.py \
  --config univ/configs/mcmae.yaml \
  --data /path/to/MVIP/FLIR-ALIGN \
  --weights univ/pretrained/checkpoint0400.pth \
  --output univ/runs/pretrain/smoke \
  --epochs 1 \
  --batch 1 \
  --seed 0
```

> 注意：该脚本是原始 RGB-IR 预训练流程，不是 M3FD detection 训练流程。

### 后续 M3FD detection smoke 命令（待检测入口适配后使用）

```bash
PYTHONUNBUFFERED=1 python -u univ/scripts/train_m3fd_univ_yolov8style_detector.py \
  --data univ/configs/m3fd.smoke.yaml \
  --modality rgb_ir \
  --fusion feature \
  --epochs 1 \
  --batch 1 \
  --imgsz 320 \
  --device 0 \
  --eval-every 1 \
  --no-freeze-backbone \
  --project univ/runs/detect/m3fd_univ_yolov8style \
  --name smoke_rgb_ir_feature_tiny_e1_no_pretrained_unfrozen \
  --seed 0 \
  2>&1 | tee univ/results/smoke_rgb_ir_feature_tiny_e1_no_pretrained_unfrozen.log
```

## 7. 正式训练流程（待 M3FD 检测入口适配后）

A. **UNIV RGB-IR feature fusion no-pretrained unfrozen**

目的：验证 UNIV 双分支结构从零训练是否能在 M3FD 上学习。

```bash
PYTHONUNBUFFERED=1 python -u univ/scripts/train_m3fd_univ_yolov8style_detector.py \
  --data univ/configs/m3fd.yaml --modality rgb_ir --fusion feature \
  --epochs 300 --batch 1 --imgsz 640 --device 0 --eval-every 1 \
  --no-freeze-backbone \
  --project univ/runs/detect/m3fd_univ_yolov8style \
  --name m3fd_univ_rgb_ir_feature_e300_no_pretrained_unfrozen_b1_eval1 \
  --seed 0
```

B. **UNIV RGB-IR feature fusion pretrained unfrozen**

目的：验证 UNIV 原始预训练权重对 M3FD RGB-IR 检测是否有帮助。

```bash
PYTHONUNBUFFERED=1 python -u univ/scripts/train_m3fd_univ_yolov8style_detector.py \
  --data univ/configs/m3fd.yaml --modality rgb_ir --fusion feature \
  --epochs 300 --batch 1 --imgsz 640 --device 0 --eval-every 1 \
  --univ-weights univ/pretrained/checkpoint0400.pth \
  --no-freeze-backbone \
  --project univ/runs/detect/m3fd_univ_yolov8style \
  --name m3fd_univ_rgb_ir_feature_e300_pretrained_unfrozen_b1_eval1 \
  --seed 0
```

C. **UNIV RGB-IR feature fusion pretrained frozen**

目的：验证固定 UNIV encoder 的表征能力。

```bash
PYTHONUNBUFFERED=1 python -u univ/scripts/train_m3fd_univ_yolov8style_detector.py \
  --data univ/configs/m3fd.yaml --modality rgb_ir --fusion feature \
  --epochs 300 --batch 1 --imgsz 640 --device 0 --eval-every 1 \
  --univ-weights univ/pretrained/checkpoint0400.pth \
  --freeze-backbone \
  --project univ/runs/detect/m3fd_univ_yolov8style \
  --name m3fd_univ_rgb_ir_feature_e300_pretrained_frozen_b1_eval1 \
  --seed 0
```

## 8. 评估流程（待检测评估入口适配后）

```bash
PYTHONUNBUFFERED=1 python -u univ/scripts/eval_m3fd_univ_yolov8style_detector.py \
  --data univ/configs/m3fd.yaml \
  --split test \
  --weights univ/runs/detect/m3fd_univ_yolov8style/<exp>/weights/best.pth \
  --output univ/results/<exp>_test.json \
  --device 0 \
  --batch 1 \
  --imgsz 640
```

评估输出至少应包含 `Precision`、`Recall`、`mAP50`、`mAP50:95` 和 per-class AP。

## 9. 结果保存规范

统一保存到：

```text
univ/runs/
univ/results/
```

命名规范：

```text
m3fd_univ_rgb_ir_feature_e300_no_pretrained_unfrozen_b1_eval1
m3fd_univ_rgb_ir_feature_e300_pretrained_unfrozen_b1_eval1
m3fd_univ_rgb_ir_feature_e300_pretrained_frozen_b1_eval1
```

## 10. 常见问题排查

- **Missing split file**：确认 `meta/train.txt`、`meta/val.txt`、`meta/test.txt` 存在；运行 `python univ/scripts/check_m3fd_for_univ.py --root /home/jinlei/database/M3FD_Detection`。
- **CUDA out of memory**：降低 `--batch` 或 `--imgsz`；优先 smoke 使用 `--batch 1 --imgsz 320`。
- **timm FutureWarning**：旧代码的 `timm.models.layers/helpers/registry` 已改为 `timm.layers`、`timm.models` 和 `timm.models._builder`。
- **checkpoint key mismatch**：确认 checkpoint 是否包含 `model`、`student`、`backbone` 等 key；使用 `strict=False` 加载时记录 missing/unexpected keys。
- **no-pretrained 却 freeze backbone**：从零训练时冻结 backbone 会导致 encoder 无法学习，通常不建议组合 `no-pretrained` 与 `--freeze-backbone`。
- **loss 下降但 mAP 很低**：检查 label 类别、坐标格式、RGB/IR 文件名配对、输入尺寸与归一化。
- **GPU 利用率低但 CPU 高**：增加 dataloader workers、检查磁盘 I/O、把数据放到本地 SSD，或减小在线增强开销。

## 11. 与 YOLO baseline 的关系

- `yolo/` 是官方 YOLO 强检测 baseline。
- `univ/` 是跨模态统一表征 baseline。
- `ours/` 是后续改进方法。
- UNIV 不需要一定超过官方 YOLO；它的价值是验证跨模态 feature fusion，并为后续语义/前景对齐模块提供可复现 baseline。

## 12. 本次兼容性整理记录

- `requirements.txt` 删除固定 PyTorch/CUDA wheel 和大量环境转储依赖，仅保留 UNIV 普通 Python 依赖。
- 修复旧 `timm.models.layers` / `timm.models.helpers` / `timm.models.registry` 导入，以适配 `timm>=0.9`。
- 新增 `univ/utils/torch_compat.py`，统一处理 `torch.load(map_location=...)` 和 `weights_only` 兼容。
- 新增 `univ/utils/IR_info_richness.py`，补齐原始训练脚本引用的 IR richness helper。
- 保留原始 UNIV 模型主体结构，未修改 `yolo/` 和 `ours/`。
