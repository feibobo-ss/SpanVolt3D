from copy import deepcopy

from torch.utils.data import Dataset

from pointcept.datasets import build_dataset
from pointcept.utils.config import Config


def load_class_names(config_file):
    cfg = Config.fromfile(config_file)
    names = list(cfg.data.names)
    if len(names) != 6:
        raise ValueError(f"Expected six classes, got {names}")
    return names


def feature_channels_from_config(cfg, split):
    transforms = (cfg.data.test.test_cfg.post_transform if split == "test"
                  else cfg.data[split].transform)
    collects = [t for t in transforms if t["type"] == "Collect"]
    if len(collects) != 1:
        raise ValueError("Expected one Collect transform")
    keys = tuple(collects[0]["feat_keys"])
    if keys not in (("coord",), ("coord", "color")):
        raise ValueError(f"Unsupported feature fields: {keys}")
    return 3 * len(keys)


class LASFinalTrainValDataset(Dataset):
    """Expose the frozen Pointcept LAS transforms to PointNet++ unchanged."""

    def __init__(self, config_file, split):
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported train/val split: {split}")
        cfg = Config.fromfile(config_file)
        self.feature_channels = feature_channels_from_config(cfg, split)
        data_cfg = deepcopy(cfg.data[split])
        data_cfg.loop = 1
        self.dataset = build_dataset(data_cfg)
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        feat = sample["feat"].float()
        segment = sample["segment"].long()
        if feat.ndim != 2 or feat.shape[1] != self.feature_channels:
            raise RuntimeError(f"Expected Nx{self.feature_channels} features, got {tuple(feat.shape)}")
        return feat, segment


def build_las_final_test_dataset(config_file, split):
    cfg = Config.fromfile(config_file)
    data_cfg = deepcopy(cfg.data.test)
    data_cfg.split = split
    return build_dataset(data_cfg), cfg
