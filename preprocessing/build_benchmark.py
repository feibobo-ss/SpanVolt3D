#!/usr/bin/env python3
"""Build the frozen six-class representation from the complete historical LAS corpus.

This is an anonymous, portable wrapper around the selected historical numerical
functions. It requires all 400 manifest scenes, not an approved delivery subset.
Read docs/data_details.md for the workflow and current access scope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core import load_las_file_basic, read_las_header_basic, remap_cloud, voxel_downsample

VERSION = "las_vegetation6_final_v1"
CLASS_NAMES = ("tower", "conductor", "jumper_conductor", "insulator", "shield_wire", "vegetation")
RAW_TO_ID = {"7": 0, "5": 1, "10": 2, "8": 3, "9": 3, "6": 4, "3": 5}
KNOWN_RAW_LABELS = set(range(13))
EXPLICITLY_DROPPED = {0, 1, 2, 4, 11, 12}
VOXEL_SIZE = 0.30
TARGET_POINTS = 1_000_000
EXPECTED_COUNTS = {"train": 253, "val": 68, "test": 62, "ood_110": 11, "ood_1000": 6}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "splits/scene_manifest.csv"


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400 or len({row["scene_id"] for row in rows}) != 400:
        raise ValueError("Expected the fixed manifest with 400 unique scene IDs")
    if Counter(row["data_split"] for row in rows) != EXPECTED_COUNTS:
        raise ValueError("Unexpected fixed split counts")
    owners = {}
    for row in rows:
        owners.setdefault(row["project_group_id"], set()).add(row["data_split"])
        for field in ("native_filename", "benchmark_filename"):
            if Path(row[field]).name != row[field] or not row[field]:
                raise ValueError(f"Expected a basename in {field}")
        if row["representation"] != VERSION:
            raise ValueError("Unexpected representation")
        if not 0 <= int(row["sampling_seed"]) < 2**32:
            raise ValueError("Invalid effective sampling seed")
    if len(owners) != 13 or any(len(splits) != 1 for splits in owners.values()):
        raise ValueError("Expected 13 disjoint project groups")
    # The existing anonymous IDs preserve the historical sorted order per split.
    return sorted(rows, key=lambda r: (list(EXPECTED_COUNTS).index(r["data_split"]), r["scene_id"]))


def limit_with_effective_seed(data, target_points, effective_seed):
    """Historical limit_target_points after its path-dependent seed derivation."""
    if target_points <= 0 or data.shape[0] <= target_points:
        return data
    rng = np.random.default_rng(int(effective_seed))
    keep_indices = rng.choice(data.shape[0], size=target_points, replace=False)
    keep_indices.sort()
    return data[keep_indices]


def process_scene(source, effective_seed):
    raw = load_las_file_basic(Path(source), SimpleNamespace(scale=1.0, default_rgb=(0.0, 0.0, 0.0)))
    unknown = {int(v) for v in np.unique(raw.labels)} - KNOWN_RAW_LABELS
    if unknown:
        raise ValueError(f"Unknown native labels: {sorted(unknown)}")
    mapped = remap_cloud(raw, RAW_TO_ID)
    output = voxel_downsample(mapped, VOXEL_SIZE)
    output = limit_with_effective_seed(output, TARGET_POINTS, effective_seed)
    if output.shape[0] == 0:
        raise ValueError("No retained points")
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True, help="Complete historical native LAS corpus, named using manifest native_filename")
    parser.add_argument("--output-root", type=Path, required=True, help="New six-class data root, outside this repository")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Check manifest, input paths and LAS headers; do not read point arrays or write files")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    native = args.native_root.resolve()
    output = args.output_root.resolve()
    missing = [row["scene_id"] for row in rows if not (native / row["native_filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} native LAS files; first IDs: {missing[:5]}")
    for row in rows:
        source = native / row["native_filename"]
        offset, point_format, record_length, count, _, _ = read_las_header_basic(source)
        if point_format != 7 or count != int(row["native_points"]):
            raise ValueError(f"{row['scene_id']}: header differs from the recorded native layer")
        if source.stat().st_size < offset + count * record_length:
            raise ValueError(f"{row['scene_id']}: native file is truncated")
    if output == native or native in output.parents or output in native.parents:
        raise ValueError("Use separate native and output roots")
    existing = [row["scene_id"] for row in rows if (output / row["data_split"] / row["benchmark_filename"]).exists()]
    if existing:
        raise FileExistsError("Output scenes already exist. Validate received six-class data with verify_benchmark.py; do not preprocess it again.")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "scenes": len(rows), "splits": EXPECTED_COUNTS, "native_headers_checked": len(rows), "writes": 0, "seed_basis": "historical_build_path_inference; full rebuild equivalence not verified"}, indent=2))
        return
    for index, row in enumerate(rows, 1):
        destination = output / row["data_split"] / row["benchmark_filename"]
        array = process_scene(native / row["native_filename"], int(row["sampling_seed"]))
        if len(array) != int(row["points"]):
            raise ValueError(f"{row['scene_id']}: point count differs from frozen record")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npy.partial")
        with temporary.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256(temporary) != row["benchmark_sha256"]:
            raise ValueError(f"{row['scene_id']}: reconstructed bytes differ from the frozen benchmark; kept .partial for inspection. The historical frozen six-class NPY defines the exact paper input bytes; see DATA_ACCESS.md for current access scope.")
        os.replace(temporary, destination)
        print(f"[{index}/{len(rows)}] {row['scene_id']}: verified frozen SHA256", flush=True)
    (output / "label_mapping.json").write_text(json.dumps({"num_classes": 6, "class_names": list(CLASS_NAMES)}, indent=2) + "\n", encoding="utf-8")
    print("Completed and verified the fixed six-class representation.")


if __name__ == "__main__":
    main()
