"""IR information-richness utilities used by legacy UNIV training."""
from __future__ import annotations

import torch


def gray_value_rank(ir_gray: torch.Tensor) -> torch.Tensor:
    """Return per-image normalized gray-value richness scores.

    The original UNIV training code optionally weights IR loss with a per-image
    ranking term. This lightweight implementation uses the spatial standard
    deviation of each grayscale IR image and normalizes it within the batch.
    """
    if ir_gray.ndim < 2:
        raise ValueError("ir_gray must contain at least batch and spatial dimensions")
    flat = ir_gray.float().flatten(start_dim=1)
    score = flat.std(dim=1)
    denom = score.max().clamp_min(1e-6)
    return score / denom
