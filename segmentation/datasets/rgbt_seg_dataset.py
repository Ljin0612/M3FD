"""Configurable, lazy RGB-thermal semantic-segmentation dataset."""

from pathlib import Path
from typing import Mapping, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_MODALITIES = {"rgb", "ir", "rgb_ir"}
_SPLITS = {"train", "val", "test"}


class RGBTSegmentationDataset(Dataset):
    """Load aligned RGB, infrared and class-mask images from an MFNet/FMB config.

    ``config`` may be either the complete configuration (with a ``dataset``
    section) or the dataset section itself.  Image paths are resolved once at
    construction time; image pixels are loaded lazily in :meth:`__getitem__`.

    The output schema is deliberately identical for every ``modality``.  Both
    aligned inputs are returned so callers can switch model modalities without
    changing their batching code; ``modality`` records and validates the input
    mode selected by the caller.
    """

    def __init__(
        self,
        config: Mapping,
        split: str = "train",
        modality: str = "rgb_ir",
        imgsz: Union[int, tuple, list, None] = None,
    ):
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping containing a 'dataset' section")

        dataset_config = config.get("dataset", config)
        if not isinstance(dataset_config, Mapping):
            raise TypeError("config['dataset'] must be a mapping")
        if split not in _SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {sorted(_SPLITS)}")
        if modality not in _MODALITIES:
            raise ValueError(
                f"Unsupported modality {modality!r}; expected one of {sorted(_MODALITIES)}"
            )

        required = ("name", "root", "rgb_dir", "ir_dir", "mask_dir", f"{split}_list")
        missing_keys = [key for key in required if not dataset_config.get(key)]
        if missing_keys:
            raise ValueError(f"Missing dataset configuration value(s): {', '.join(missing_keys)}")

        self.name = str(dataset_config["name"])
        if self.name.lower() not in {"mfnet", "fmb"}:
            raise ValueError("dataset.name must be 'MFNet' or 'FMB'")
        self.root = Path(dataset_config["root"]).expanduser()
        self.modality = modality
        self.num_classes = dataset_config.get("num_classes")
        self.ignore_index = int(dataset_config.get("ignore_index", 255))
        self.class_names = list(dataset_config.get("class_names", []))
        self.image_size = self._parse_image_size(
            imgsz if imgsz is not None else dataset_config.get("imgsz")
        )

        split_path = self._under_root(dataset_config[f"{split}_list"])
        if not split_path.is_file():
            raise FileNotFoundError(f"{split} split file does not exist: {split_path}")
        sample_names = self._read_split(split_path)
        if not sample_names:
            raise ValueError(f"No samples found in {split} split file: {split_path}")

        folders = {
            "rgb": self._under_root(dataset_config["rgb_dir"]),
            "ir": self._under_root(dataset_config["ir_dir"]),
            "mask": self._under_root(dataset_config["mask_dir"]),
        }
        self.samples = []
        for sample_name in sample_names:
            paths = {}
            for kind, folder in folders.items():
                try:
                    paths[kind] = self._resolve_file(folder, sample_name)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Missing {kind.upper()} file for sample {sample_name!r} "
                        f"in {folder} (split: {split})"
                    ) from exc
            self.samples.append((sample_name, paths["rgb"], paths["ir"], paths["mask"]))

    @staticmethod
    def _parse_image_size(imgsz):
        if imgsz is None:
            return None
        if isinstance(imgsz, int):
            if imgsz <= 0:
                raise ValueError("imgsz must be positive")
            return (imgsz, imgsz)
        if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            height, width = (int(value) for value in imgsz)
            if height <= 0 or width <= 0:
                raise ValueError("imgsz dimensions must be positive")
            return (height, width)
        raise TypeError("imgsz must be a positive integer or a (height, width) pair")

    def _under_root(self, path_value):
        path = Path(path_value).expanduser()
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _read_split(split_path):
        samples = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                samples.append(line.split()[0])
        return samples

    @staticmethod
    def _resolve_file(folder, sample_name):
        relative = Path(sample_name)
        candidate = folder / relative
        if candidate.is_file():
            return candidate
        # A listed suffix may be specific to one modality.  Try the same stem
        # with all common image suffixes before declaring the sample missing.
        stem = relative.with_suffix("") if relative.suffix else relative
        for suffix in _IMAGE_SUFFIXES:
            candidate = folder / stem.parent / f"{stem.name}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(candidate)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_name, rgb_path, ir_path, mask_path = self.samples[index]
        # Conversion occurs inside each context manager, detaching the returned
        # PIL image from the underlying file before its descriptor is closed.
        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB")
        with Image.open(ir_path) as image:
            ir = image.convert("L")
        with Image.open(mask_path) as image:
            mask = image.copy()

        if rgb.size != ir.size or rgb.size != mask.size:
            raise ValueError(
                f"RGB, IR and mask sizes differ for sample {sample_name!r}: "
                f"rgb={rgb.size}, ir={ir.size}, mask={mask.size}"
            )
        if self.image_size is not None:
            height, width = self.image_size
            pil_size = (width, height)
            rgb = rgb.resize(pil_size, Image.Resampling.BILINEAR)
            ir = ir.resize(pil_size, Image.Resampling.BILINEAR)
            mask = mask.resize(pil_size, Image.Resampling.NEAREST)

        rgb_array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1).copy()
        ir_array = np.asarray(ir, dtype=np.float32)[None, ...].copy()
        mask_array = np.asarray(mask, dtype=np.int64).copy()
        return {
            "rgb": torch.from_numpy(rgb_array).div_(255.0),
            "ir": torch.from_numpy(ir_array).div_(255.0),
            "mask": torch.from_numpy(mask_array).long(),
            "path": sample_name,
        }
