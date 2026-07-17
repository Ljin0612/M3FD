"""Small, consistent checkpoint helpers."""

from pathlib import Path

import torch


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **metadata}, path)


def load_checkpoint(path, model, map_location="cpu", strict=True):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload.get("model", payload), strict=strict)
    return payload
