#!/usr/bin/env python3
"""Compatibility entry point for segmentation evaluation."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("evaluate.py")), run_name="__main__")
