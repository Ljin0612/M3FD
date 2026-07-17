#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.engine import make_loader, run_epoch
from segmentation.models import build_model

parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
model = build_model(config).to(device)
state = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state.get("model", state))
stats = run_epoch(model, make_loader(config, "val_split"), device, config["num_classes"], config.get("ignore_index", 255))
print(json.dumps(stats, indent=2))
