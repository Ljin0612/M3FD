"""Segmentation models and RGB-IR baselines."""
from .factory import build_model
from .unet import UNet, UNetBaseline

__all__ = ["UNet", "UNetBaseline", "build_model"]
