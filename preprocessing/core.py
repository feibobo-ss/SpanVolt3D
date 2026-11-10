"""Selected historical LAS reading, remapping and sampling functions.

Numerical function bodies are preserved from the original preprocessing module.
"""
import hashlib
import math
import struct
import numpy as np


def numeric_label_key(label):
    try:
        value = float(label)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not np.isclose(value, round(value)):
        return None
    return int(round(value))


def normalize_label_token(label):
    text = str(label).strip()
    if text == "":
        raise ValueError("empty label")
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value) or not np.isclose(value, round(value)):
        raise ValueError(f"numeric labels must be integer ids, got {label!r}")
    return str(int(round(value)))


def read_las_header_basic(path):
    with path.open("rb") as fin:
        header = fin.read(375)
    if len(header) < 227 or header[:4] != b"LASF":
        raise ValueError(f"{path}: not a valid LAS file.")

    header_size = struct.unpack_from("<H", header, 94)[0]
    point_data_offset = struct.unpack_from("<I", header, 96)[0]
    point_format_raw = header[104]
    point_format = point_format_raw & 0x3F
    record_length = struct.unpack_from("<H", header, 105)[0]
    point_count = struct.unpack_from("<I", header, 107)[0]

    if point_count == 0 and len(header) >= 255 and header_size >= 375:
        point_count = struct.unpack_from("<Q", header, 247)[0]

    scales = struct.unpack_from("<ddd", header, 131)
    offsets = struct.unpack_from("<ddd", header, 155)
    if point_count <= 0:
        raise ValueError(f"{path}: no point records found in LAS header.")
    return point_data_offset, point_format, record_length, int(point_count), scales, offsets


def las_field_offsets(point_format):
    if point_format in {0, 1, 2, 3, 4, 5}:
        class_offset = 15
    elif point_format in {6, 7, 8, 9, 10}:
        class_offset = 16
    else:
        raise ValueError(f"unsupported LAS point format {point_format}")

    rgb_offset = None
    if point_format == 2:
        rgb_offset = 20
    elif point_format in {3, 5}:
        rgb_offset = 28
    elif point_format in {7, 8, 10}:
        rgb_offset = 30
    return class_offset, rgb_offset


def normalize_las_rgb(rgb):
    if not rgb.size:
        return rgb
    max_rgb = float(np.nanmax(rgb))
    if max_rgb <= 255:
        return rgb
    if max_rgb > 4096:
        return rgb / 65535.0 * 255.0
    return rgb / max_rgb * 255.0


def load_las_file_basic(path, args):
    point_data_offset, point_format, record_length, point_count, scales, offsets = read_las_header_basic(path)
    class_offset, rgb_offset = las_field_offsets(point_format)

    names = ["x", "y", "z", "classification"]
    formats = ["<i4", "<i4", "<i4", "u1"]
    field_offsets = [0, 4, 8, class_offset]
    if rgb_offset is not None:
        names += ["red", "green", "blue"]
        formats += ["<u2", "<u2", "<u2"]
        field_offsets += [rgb_offset, rgb_offset + 2, rgb_offset + 4]

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": field_offsets,
        "itemsize": record_length,
    })
    records = np.memmap(path, mode="r", dtype=dtype, offset=point_data_offset, shape=(point_count,))

    x = records["x"].astype(np.float64) * scales[0] + offsets[0]
    y = records["y"].astype(np.float64) * scales[1] + offsets[1]
    z = records["z"].astype(np.float64) * scales[2] + offsets[2]
    xyz = np.column_stack([x, y, z]).astype(np.float32) * args.scale

    if rgb_offset is None:
        rgb = np.tile(np.asarray(args.default_rgb, dtype=np.float32), (point_count, 1))
    else:
        rgb = np.column_stack([records["red"], records["green"], records["blue"]]).astype(np.float32)
        rgb = normalize_las_rgb(rgb)
    rgb = np.clip(rgb, 0, 255)

    labels = np.asarray(records["classification"], dtype=np.int16)
    return RawCloud(path=path, xyz=xyz, rgb=rgb, labels=labels, skipped_rows=0)


class RawCloud:
    def __init__(self, path, xyz, rgb, labels, skipped_rows=0):
        self.path = path
        self.xyz = xyz
        self.rgb = rgb
        self.labels = labels
        self.skipped_rows = skipped_rows

    @property
    def point_count(self):
        return len(self.labels)


def build_numeric_lookup(label_to_id):
    numeric_items = []
    for original_label, new_id in label_to_id.items():
        numeric_label = numeric_label_key(original_label)
        if numeric_label is None:
            return None, None
        numeric_items.append((numeric_label, new_id))

    if not numeric_items:
        return None, None

    min_label = min(label for label, _ in numeric_items)
    max_label = max(label for label, _ in numeric_items)
    span = max_label - min_label + 1
    if span > max(1_000_000, len(numeric_items) * 1000):
        return None, None

    lookup = np.full(span, -1, dtype=np.int32)
    for original_label, new_id in numeric_items:
        lookup[original_label - min_label] = int(new_id)
    return lookup, min_label


def remap_labels(labels, label_to_id):
    if isinstance(labels, np.ndarray) and labels.dtype.kind in {"i", "u"}:
        lookup, offset = build_numeric_lookup(label_to_id)
        if lookup is not None:
            label_values = labels.astype(np.int64, copy=False)
            shifted = label_values - offset
            keep = (shifted >= 0) & (shifted < lookup.shape[0])
            mapped = np.full(label_values.shape, -1, dtype=np.int32)
            if keep.any():
                mapped[keep] = lookup[shifted[keep]]
            keep &= mapped >= 0
            return mapped[keep].astype(np.float32, copy=False).reshape(-1, 1), keep

    mapped_labels = []
    keep_mask = []
    for label in labels:
        key = normalize_label_token(label)
        if key in label_to_id:
            mapped_labels.append(label_to_id[key])
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return np.asarray(mapped_labels, dtype=np.float32).reshape(-1, 1), np.asarray(keep_mask, dtype=bool)


def remap_cloud(cloud, label_to_id):
    label, keep_mask = remap_labels(cloud.labels, label_to_id)
    if label.shape[0] == 0:
        return np.empty((0, 7), dtype=np.float32)
    xyz = cloud.xyz[keep_mask]
    rgb = cloud.rgb[keep_mask]
    return np.concatenate([xyz, rgb, label], axis=1).astype(np.float32, copy=False)


def voxel_downsample(data, voxel_size):
    if voxel_size <= 0 or data.shape[0] == 0:
        return data
    voxel_coords = np.floor(data[:, :3] / voxel_size).astype(np.int64, copy=False)
    _, keep_indices = np.unique(voxel_coords, axis=0, return_index=True)
    keep_indices.sort()
    return data[keep_indices]


def stable_seed(seed, path):
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).digest()
    path_seed = int.from_bytes(digest, "little")
    return (int(seed) + path_seed) % (2 ** 32)


def limit_target_points(data, target_points, seed, source_path):
    if target_points <= 0 or data.shape[0] <= target_points:
        return data
    rng = np.random.default_rng(stable_seed(seed, source_path))
    keep_indices = rng.choice(data.shape[0], size=target_points, replace=False)
    keep_indices.sort()
    return data[keep_indices]
