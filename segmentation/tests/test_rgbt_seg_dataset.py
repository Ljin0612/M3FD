from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.datasets import RGBTSegmentationDataset


def _make_config(tmp_path):
    for directory in ("visible", "thermal", "labels", "splits"):
        (tmp_path / directory).mkdir()
    (tmp_path / "splits" / "train.txt").write_text("nested/example\n", encoding="utf-8")
    for directory in ("visible", "thermal", "labels"):
        (tmp_path / directory / "nested").mkdir()

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, 1:] = 255
    ir = np.array([[0, 127, 255], [255, 127, 0]], dtype=np.uint8)
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "visible" / "nested" / "example.jpg")
    Image.fromarray(ir).save(tmp_path / "thermal" / "nested" / "example.png")
    Image.fromarray(mask).save(tmp_path / "labels" / "nested" / "example.png")
    return {
        "dataset": {
            "name": "MFNet",
            "root": tmp_path,
            "rgb_dir": "visible",
            "ir_dir": "thermal",
            "mask_dir": "labels",
            "train_list": "splits/train.txt",
            "num_classes": 3,
            "ignore_index": 255,
            "class_names": ["background", "one", "two"],
        }
    }


@pytest.mark.parametrize("modality", ["rgb", "ir", "rgb_ir"])
def test_config_modalities_and_synchronized_resize(tmp_path, modality):
    dataset = RGBTSegmentationDataset(
        _make_config(tmp_path), split="train", modality=modality, imgsz=4
    )

    sample = dataset[0]
    assert set(sample) == {"rgb", "ir", "mask", "path"}
    assert sample["rgb"].shape == (3, 4, 4)
    assert sample["ir"].shape == (1, 4, 4)
    assert sample["mask"].shape == (4, 4)
    assert sample["rgb"].dtype == sample["ir"].dtype == torch.float32
    assert sample["mask"].dtype == torch.long
    assert set(sample["mask"].unique().tolist()) <= {0, 1, 2}
    assert sample["path"] == "nested/example"


def test_missing_modality_has_specific_error(tmp_path):
    config = _make_config(tmp_path)
    Path(tmp_path / "thermal" / "nested" / "example.png").unlink()

    with pytest.raises(FileNotFoundError, match="Missing IR file.*nested/example"):
        RGBTSegmentationDataset(config)


def test_split_and_modality_are_validated(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Unsupported split"):
        RGBTSegmentationDataset(config, split="dev")
    with pytest.raises(ValueError, match="Unsupported modality"):
        RGBTSegmentationDataset(config, modality="depth")
