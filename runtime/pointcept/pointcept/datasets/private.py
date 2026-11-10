"""
Private power-line semantic segmentation dataset.

Input files are the repository-level processed_data/*.npy files with columns:
    x, y, z, r, g, b, label
"""

import glob
import os
from collections.abc import Sequence

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class PrivateDataset(DefaultDataset):
    # Identity mapping for private semantic taxonomies.  Keep enough entries for
    # both the frozen six-class benchmark and derived application label spaces.
    class2id = np.arange(256)

    def __init__(self, use_normalized_coord=True, file_names=None, use_color=True, **kwargs):
        self.use_normalized_coord = use_normalized_coord
        self.file_names = file_names
        # Keep historical XYZ+RGB behavior by default. XYZ-only experiments
        # omit color entirely, including from the augmentation input dictionary.
        self.use_color = use_color
        super().__init__(**kwargs)
        self.name_to_path = {self.get_name_from_path(path): path for path in self.data_list}

    @staticmethod
    def get_name_from_path(path):
        return os.path.splitext(os.path.basename(path))[0]

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            split_path = os.path.join(self.data_root, split)
            data_list += glob.glob(os.path.join(split_path, "*.npy"))
        data_list = sorted(data_list)
        if self.file_names is not None:
            names = (
                {self.file_names}
                if isinstance(self.file_names, str)
                else set(self.file_names)
            )
            names = {os.path.splitext(name)[0] for name in names}
            data_list = [
                path for path in data_list if self.get_name_from_path(path) in names
            ]
        if len(data_list) == 0:
            raise FileNotFoundError(
                f"No .npy files found under {self.data_root} for split {self.split}"
            )
        return data_list

    def get_data_name(self, idx):
        return self.get_name_from_path(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(os.path.dirname(self.data_list[idx % len(self.data_list)]))

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        raw = np.load(data_path, mmap_mode=None if self.use_color else "r")
        if raw.ndim != 2 or raw.shape[1] < 7:
            raise ValueError(f"Expected Nx7 array in {data_path}, got {raw.shape}")

        coord = raw[:, 0:3].astype(np.float32)
        segment = raw[:, 6].reshape(-1).astype(np.int32)

        if self.use_normalized_coord:
            coord_min = coord.min(axis=0)
            coord_range = coord.max(axis=0) - coord_min
            coord_range[coord_range == 0] = 1.0
            normal = ((coord - coord_min) / coord_range).astype(np.float32)
        else:
            normal = np.zeros_like(coord, dtype=np.float32)

        data = dict(
            coord=coord,
            normal=normal,
            segment=segment,
            name=self.get_name_from_path(data_path),
            split=os.path.basename(os.path.dirname(data_path)),
        )
        if self.use_color:
            data["color"] = np.clip(raw[:, 3:6], 0, 255).astype(np.float32)
        return data

    def get_raw_data_by_name(self, name):
        if name not in self.name_to_path:
            raise KeyError(f"{name} not found in PrivateDataset split {self.split}")
        return np.load(self.name_to_path[name])
