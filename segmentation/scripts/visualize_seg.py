#!/usr/bin/env python3
"""Colorize a class-index mask for quick dataset inspection."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.utils.palette import colorize_mask

parser = argparse.ArgumentParser()
parser.add_argument("mask")
parser.add_argument("output")
args = parser.parse_args()
colorize_mask(np.asarray(Image.open(args.mask))).save(args.output)
