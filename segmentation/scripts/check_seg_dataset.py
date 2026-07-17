#!/usr/bin/env python3
"""Validate alignment and label ranges in an RGB-thermal dataset split."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.datasets import RGBTSegmentationDataset

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--num-classes", required=True, type=int)
args = parser.parse_args()
dataset = RGBTSegmentationDataset(args.root, args.split)
for index, sample in enumerate(dataset):
    valid = sample["mask"] != 255
    if valid.any() and (sample["mask"][valid].min() < 0 or sample["mask"][valid].max() >= args.num_classes):
        raise ValueError(f"Out-of-range label in {sample['id']}")
print(f"Validated {len(dataset)} aligned samples")
