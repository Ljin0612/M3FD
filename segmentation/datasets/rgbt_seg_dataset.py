"""Configurable RGB-thermal segmentation dataset."""

from .paired import PairedSegmentationDataset


class RGBTSegmentationDataset(PairedSegmentationDataset):
    """Paired dataset with configurable modality directory names."""

    def __init__(self, *args, rgb_dir="rgb", ir_dir="ir", mask_dir="masks", **kwargs):
        folders = {"rgb": rgb_dir, "ir": ir_dir, "mask": mask_dir}
        super().__init__(*args, folders=folders, **kwargs)
