import torch

from segmentation.models import build_model
from segmentation.utils import SegmentationMetrics


def _config(name):
    return {"num_classes": 3, "model": {"name": name, "base_channels": 4},
            "data": {"image_size": [32, 32]}}


def test_all_unet_modalities_preserve_spatial_shape():
    rgb, ir = torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32)
    for name in ("unet_rgb", "unet_ir", "unet_early_fusion"):
        model = build_model(_config(name)).eval()
        with torch.no_grad():
            assert model(rgb, ir).shape == (1, 3, 32, 32)


def test_metrics_ignore_void_pixels():
    logits = torch.tensor([[[[5.0, 0.0]], [[0.0, 5.0]]]])
    target = torch.tensor([[[0, 255]]])
    metrics = SegmentationMetrics(2)
    metrics.update(logits, target)
    result = metrics.compute()
    assert result["pixel_accuracy"] == 1.0
    assert result["mean_iou"] == 1.0
