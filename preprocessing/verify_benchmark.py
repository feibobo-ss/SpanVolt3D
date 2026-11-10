#!/usr/bin/env python3
"""Validate received frozen six-class arrays without repeating preprocessing."""
import argparse
from pathlib import Path
import numpy as np
from build_benchmark import DEFAULT_MANIFEST, EXPECTED_COUNTS, read_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sha256", action="store_true", help="Additionally read every array's bytes and compare the recorded historical SHA256")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    for split in EXPECTED_COUNTS:
        actual = {p.name for p in (args.data_root / split).glob("*.npy")}
        expected = {r["benchmark_filename"] for r in rows if r["data_split"] == split}
        if actual != expected:
            raise ValueError(f"{split}: missing={len(expected-actual)}, unexpected={len(actual-expected)}")
    for row in rows:
        path = args.data_root / row["data_split"] / row["benchmark_filename"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (int(row["points"]), 7) or array.dtype != np.dtype("float32"):
            raise ValueError(f"Unexpected shape/dtype: {row['scene_id']}")
        if args.sha256 and sha256(path) != row["benchmark_sha256"]:
            raise ValueError(f"Historical SHA256 mismatch: {row['scene_id']}")
    print(f"Verified {len(rows)} files: fixed membership, N×7 shape, float32 dtype, point count" + (", historical SHA256." if args.sha256 else ". Content not fully scanned; add --sha256 for byte verification."))


if __name__ == "__main__":
    main()
