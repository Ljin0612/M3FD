"""Small PyTorch compatibility helpers for legacy UNIV checkpoints."""
from __future__ import annotations

import inspect
from typing import Any

import torch


def torch_load(path: str, map_location: str | torch.device = "cpu", weights_only: bool | None = None) -> Any:
    """Load a checkpoint with a stable map_location and optional weights_only support.

    PyTorch 2.4+ exposes ``weights_only`` on ``torch.load`` and may warn about
    the default value. Older PyTorch releases do not accept that keyword. This
    wrapper keeps UNIV checkpoint loading compatible across both versions.
    """
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None and "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)
