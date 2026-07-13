"""Train UNIV-M3FD RGB-IR YOLOv8-style detector."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from univ.scripts.datasets.m3fd_rgbt_detection_dataset import M3FDRGBTDetectionDataset
from univ.scripts.m3fd_train_eval_common import append_csv, collate_m3fd, count_parameters, cuda_memory_string, ensure_dir, load_yaml, set_seed, simple_detection_loss, write_kv, NC
from univ.scripts.models.m3fd_univ_rgbt_yolov8style_detector import M3FDUNIVRGBTYOLOv8StyleDetector, load_univ_checkpoint


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
    fields = ["epoch", "batch_idx", "loss_total", "loss_box", "loss_obj", "loss_cls"]

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

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for batch_idx, b in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            preds = model(b["visible"].to(device), b["infrared"].to(device))
            losses = simple_detection_loss(preds, b["targets"], NC)
            losses["loss_total"].backward()
            opt.step()
            row = {k: (float(losses[k].detach().cpu()) if k.startswith("loss") else locals()[k]) for k in fields}
            append_csv(out / "train_log.csv", row, fields)
            if batch_idx % 20 == 0:
                print(f"epoch={epoch} batch_idx={batch_idx} loss_total={row['loss_total']:.4f} loss_box={row['loss_box']:.4f} loss_obj={row['loss_obj']:.4f} loss_cls={row['loss_cls']:.4f} {cuda_memory_string()}")
        last_loss = row["loss_total"]
        ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "load_report": report.__dict__}
        torch.save(ckpt, out / "last.pth")
        if last_loss < best:
            best = last_loss
            torch.save(ckpt, out / "best.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
