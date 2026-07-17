"""Segmentation models and RGB-IR baselines."""
from .factory import build_model
from .unet import UNet, UNetBaseline
from .rgbir_early_fusion import RGBIREarlyFusionUNet

__all__ = ["UNet", "UNetBaseline", "RGBIREarlyFusionUNet", "build_model"]
