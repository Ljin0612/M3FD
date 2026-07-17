"""Compact U-Net baselines for RGB, IR, and early fusion."""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=32):
        super().__init__()
        channels = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([DoubleConv(in_channels, channels[0])] +
                                      [DoubleConv(channels[i - 1], channels[i]) for i in range(1, 5)])
        self.decoders = nn.ModuleList([DoubleConv(channels[i] + channels[i - 1], channels[i - 1])
                                      for i in range(4, 0, -1)])
        self.classifier = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x):
        features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else F.max_pool2d(x, 2))
            features.append(x)
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.classifier(x)


class UNetBaseline(nn.Module):
    """Select the requested input modality before applying U-Net."""
    def __init__(self, modality, num_classes, base_channels=32):
        super().__init__()
        if modality not in {"rgb", "ir", "early_fusion"}:
            raise ValueError(f"Unsupported U-Net modality: {modality}")
        self.modality = modality
        self.unet = UNet({"rgb": 3, "ir": 1, "early_fusion": 4}[modality], num_classes, base_channels)

    def forward(self, rgb, ir):
        x = {"rgb": rgb, "ir": ir}.get(self.modality)
        if x is None:
            x = torch.cat((rgb, ir), dim=1)
        return self.unet(x)
