from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from segmentation.datasets import PairedSegmentationDataset
from segmentation.models import build_model
from segmentation.utils import SegmentationMetrics


def make_loader(config, split, training=False):
    data = config["data"]
    dataset = PairedSegmentationDataset(data["root"], data[split], data.get("image_size"),
                                        augment=training, ignore_index=config.get("ignore_index", 255))
    return DataLoader(dataset, batch_size=config["training"].get("batch_size", 4),
                      shuffle=training, num_workers=config["training"].get("workers", 2))


def run_epoch(model, loader, device, num_classes, ignore_index, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes, ignore_index)
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    for batch in loader:
        rgb, ir, target = (batch[key].to(device) for key in ("rgb", "ir", "mask"))
        with torch.set_grad_enabled(training):
            logits = model(rgb, ir)
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * target.shape[0]
        metrics.update(logits, target)
    result = metrics.compute()
    result["loss"] = total_loss / len(loader.dataset)
    return result


def train(config, output_dir):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"].get("weight_decay", 1e-4))
    train_loader = make_loader(config, "train_split", True)
    val_loader = make_loader(config, "val_split")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(config["training"]["epochs"]):
        train_stats = run_epoch(model, train_loader, device, config["num_classes"], config.get("ignore_index", 255), optimizer)
        val_stats = run_epoch(model, val_loader, device, config["num_classes"], config.get("ignore_index", 255))
        print(json.dumps({"epoch": epoch + 1, "train": train_stats, "val": val_stats}))
        if val_stats["mean_iou"] > best:
            best = val_stats["mean_iou"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch + 1}, output / "best.pth")
    return model
