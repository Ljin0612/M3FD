"""RGB-IR channel-concatenation baseline."""

from .unet import UNetBaseline


class RGBIREarlyFusionUNet(UNetBaseline):
    def __init__(self, num_classes, base_channels=32):
        super().__init__("early_fusion", num_classes, base_channels)
