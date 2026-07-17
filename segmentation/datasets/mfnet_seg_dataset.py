"""MFNet semantic-segmentation dataset adapter."""

from .rgbt_seg_dataset import RGBTSegmentationDataset


class MFNetSegmentationDataset(RGBTSegmentationDataset):
    """MFNet loader; directory names may still be overridden when converted."""

    pass
