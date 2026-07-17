#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.engine import train

parser = argparse.ArgumentParser(description="Train an RGB-IR segmentation baseline")
parser.add_argument("--config", required=True)
parser.add_argument("--output", default="segmentation/runs/experiment")
args = parser.parse_args()
with open(args.config, encoding="utf-8") as stream:
    train(yaml.safe_load(stream), args.output)
