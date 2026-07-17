"""Loss construction for semantic segmentation."""

from torch import nn


def build_segmentation_loss(ignore_index=255, class_weights=None):
    """Build cross-entropy with the repository's standard void-label handling."""
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
