"""UNIV-original encoder with a lightweight semantic segmentation head."""

from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F


class UNIVSegmentation(nn.Module):
    def __init__(self, num_classes, image_size=224, checkpoint=None, freeze_encoder=False):
        super().__init__()
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root / "univ") not in sys.path:
            sys.path.insert(0, str(repo_root / "univ"))
        from models.backbone.mcmae.models_convmae import convmae_convvit_base_patch16
        self.encoder = convmae_convvit_base_patch16(img_size=[image_size, image_size // 4, image_size // 8])
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = state.get("model", state.get("state_dict", state))
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected UNIV checkpoint keys: {unexpected[:5]}")
        self.head = nn.Sequential(nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, num_classes, 1))
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def forward(self, rgb, ir):
        # UNIV pretraining consumes three-channel views. IR is repeated and the
        # original encoder is shared, then modality tokens are averaged.
        ir_rgb = ir.repeat(1, 3, 1, 1)
        rgb_tokens, _ = self.encoder(rgb, mask_ratio=0.0)
        ir_tokens, _ = self.encoder(ir_rgb, mask_ratio=0.0)
        tokens = (rgb_tokens + ir_tokens) * 0.5
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("UNIV token sequence is not a square spatial grid")
        logits = self.head(tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side))
        return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
