"""Device-memory reporting without requiring CUDA."""

import torch


def cuda_memory_megabytes():
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    scale = 1024 ** 2
    return {"allocated_mb": torch.cuda.memory_allocated() / scale,
            "reserved_mb": torch.cuda.memory_reserved() / scale}
