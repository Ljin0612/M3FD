"""Deterministic colors and mask colorization."""

import numpy as np
from PIL import Image


def voc_palette(size=256):
    palette = []
    for label in range(size):
        red = green = blue = 0
        value = label
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend((red, green, blue))
    return palette


def colorize_mask(mask):
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="P")
    image.putpalette(voc_palette())
    return image
