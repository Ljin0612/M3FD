"""Dataset definitions for RGB-IR semantic segmentation."""
from .paired import PairedSegmentationDataset
from .rgbt_seg_dataset import RGBTSegmentationDataset
from .mfnet_seg_dataset import MFNetSegmentationDataset
from .fmb_seg_dataset import FMBSegmentationDataset

__all__ = ["PairedSegmentationDataset", "RGBTSegmentationDataset",
           "MFNetSegmentationDataset", "FMBSegmentationDataset"]
