"""FMB semantic-segmentation dataset adapter."""

from .rgbt_seg_dataset import RGBTSegmentationDataset


class FMBSegmentationDataset(RGBTSegmentationDataset):
    """FMB loader; directory names may still be overridden when converted."""

    pass
