"""Train UNIV-M3FD RGB-IR YOLOv8-style detector."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

if importlib.util.find_spec("cv2") is not None:
    import cv2

    cv2.setNumThreads(0)

import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from univ.scripts.datasets.m3fd_rgbt_detection_dataset import M3FDRGBTDetectionDataset
from univ.scripts.m3fd_train_eval_common import append_csv, collate_m3fd, count_parameters, ensure_dir, load_yaml, set_seed, simple_detection_loss, write_kv, NC
from univ.scripts.models.m3fd_univ_rgbt_yolov8style_detector import M3FDUNIVRGBTYOLOv8StyleDetector, load_univ_checkpoint


def memory_stats(process: psutil.Process, device: torch.device) -> dict[str, float]:
    cpu_rss_mb = process.memory_info().rss / (1024 ** 2)
    cuda_allocated_mb = 0.0
    cuda_reserved_mb = 0.0
    if device.type == "cuda":
        cuda_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        cuda_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return {
        "cpu_rss_mb": cpu_rss_mb,
        "cuda_allocated_mb": cuda_allocated_mb,
        "cuda_reserved_mb": cuda_reserved_mb,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, best_loss: float, args: argparse.Namespace, cfg: dict, report) -> None:
    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "model_state_dict": model_state_dict,
        "epoch": epoch,
        "best_loss": best_loss,
        "args": vars(args),
        "config": cfg,
        "load_report": report.__dict__,
    }
    torch.save(ckpt, path)
    del ckpt, model_state_dict
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="univ/configs/m3fd.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--univ-weights", default=None)
    p.add_argument("--checkpoint-key", choices=["student", "teacher"], default="student")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--modality", default=None)
    p.add_argument("--fusion", default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--memory-log-every", type=int, default=100)
    p.add_argument("--empty-cache-every", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    data = args.data or cfg.get("data") or cfg.get("dataset", {}).get("path")
    if args.data and Path(args.data).suffix.lower() in {".yaml", ".yml"}:
        cfg = {**cfg, **load_yaml(args.data)}
    imgsz = args.imgsz or int(cfg.get("imgsz", 224))
    batch = args.batch or int(cfg.get("batch", 4))
    epochs = args.epochs or int(cfg.get("epochs", 1))
    lr = args.lr or float(cfg.get("lr", 1e-4))
    workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 2))
    modality = args.modality or cfg.get("modality", "rgb-ir")
    fusion = args.fusion or cfg.get("fusion", "concat")
    set_seed(args.seed if args.seed is not None else int(cfg.get("seed", 7)))
    device_arg = args.device
    if device_arg and device_arg.isdigit():
        device_arg = f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir or (str(Path(args.project) / args.name) if args.project and args.name else "univ/runs/m3fd_univ_detector")
    out = ensure_dir(output_dir)

    model = M3FDUNIVRGBTYOLOv8StyleDetector(nc=NC, imgsz=imgsz, shared_encoder=True, fusion=fusion)
    report = load_univ_checkpoint(model, args.univ_weights, args.checkpoint_key)
    freeze_final = bool(args.freeze_backbone)
    if args.freeze_backbone and not report.weights_loaded:
        print("WARNING: freezing randomly initialized UNIV encoder is not a valid train-from-scratch baseline.")
    if freeze_final:
        model.freeze_backbone()
    model.to(device)

    ds = M3FDRGBTDetectionDataset(data, split="train", imgsz=imgsz, modality=modality)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=collate_m3fd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    process = psutil.Process(os.getpid())
    fields = [
        "epoch",
        "batch_idx",
        "loss_total",
        "loss_box",
        "loss_obj",
        "loss_cls",
        "cpu_rss_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
    ]

    first = next(iter(dl))
    with torch.no_grad():
        pred0 = model(first["visible"].to(device), first["infrared"].to(device))
    trainable, frozen = count_parameters(model)
    write_kv(out / "train_meta.txt", {
        "data path": data, "modality": modality, "fusion": fusion, "imgsz": imgsz, "batch": batch, "epochs": epochs,
        "univ_weights path": args.univ_weights, "weights_exists": report.weights_exists, "weights_loaded": report.weights_loaded,
        "checkpoint_key": report.checkpoint_key, "loaded_keys count": report.loaded_keys, "skipped_keys count": report.skipped_keys,
        "interpolated_keys count": report.interpolated_keys, "missing_keys count": report.missing_keys, "unexpected_keys count": report.unexpected_keys,
        "original pos_embed shape": report.original_pos_embed_shape, "final pos_embed shape": report.final_pos_embed_shape,
        "requested imgsz": report.requested_imgsz, "encoder token grid size": report.encoder_token_grid_size,
        "freeze_backbone final value": freeze_final, "trainable parameter count": trainable, "frozen parameter count": frozen,
        "visible input shape": tuple(first["visible"].shape), "infrared input shape": tuple(first["infrared"].shape),
        "visible feature shape": getattr(model, "last_visible_feature_shape", None), "infrared feature shape": getattr(model, "last_infrared_feature_shape", None),
        "fused feature shape": getattr(model, "last_fused_feature_shape", None), "fuse input channels": model.fuse_input_channels,
        "fuse output channels": model.fuse_output_channels,
        "prediction shape": [tuple(p.shape) for p in pred0],
    })
    del pred0, first
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for batch_idx, b in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            preds = model(b["visible"].to(device), b["infrared"].to(device))
            losses = simple_detection_loss(preds, b["targets"], NC)
            losses["loss_total"].backward()
            opt.step()
            row = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "loss_total": float(losses["loss_total"].detach().cpu()),
                "loss_box": float(losses["loss_box"].detach().cpu()),
                "loss_obj": float(losses["loss_obj"].detach().cpu()),
                "loss_cls": float(losses["loss_cls"].detach().cpu()),
            }
            if args.memory_log_every > 0 and batch_idx % args.memory_log_every == 0:
                row.update(memory_stats(process, device))
            else:
                row.update({"cpu_rss_mb": "", "cuda_allocated_mb": "", "cuda_reserved_mb": ""})
            append_csv(out / "train_log.csv", row, fields)
            if args.log_every > 0 and batch_idx % args.log_every == 0:
                msg = (
                    f"epoch={epoch} batch_idx={batch_idx} loss_total={row['loss_total']:.4f} "
                    f"loss_box={row['loss_box']:.4f} loss_obj={row['loss_obj']:.4f} loss_cls={row['loss_cls']:.4f}"
                )
                if row["cpu_rss_mb"] != "":
                    msg += (
                        f" cpu_rss_mb={row['cpu_rss_mb']:.1f}"
                        f" cuda_allocated_mb={row['cuda_allocated_mb']:.1f}"
                        f" cuda_reserved_mb={row['cuda_reserved_mb']:.1f}"
                    )
                print(msg)
            del preds, losses, b
            if args.empty_cache_every > 0 and (batch_idx + 1) % args.empty_cache_every == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
        last_loss = row["loss_total"]
        if last_loss < best:
            best = last_loss
            save_checkpoint(out / "best.pth", model, epoch, best, args, cfg, report)
        save_checkpoint(out / "last.pth", model, epoch, best, args, cfg, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
