"""Evaluate existing six-class predictions/counts without running a model."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from historical_metrics import (
    compute_metrics, confusion_from_arrays, intersection_and_union, pointcept_metrics,
)
from protocol import NAMES


def validated_confusion(value):
    value = np.asarray(value)
    if value.shape != (6, 6) or not np.isfinite(value).all():
        raise ValueError("A confusion matrix must be finite and have shape (6, 6)")
    if (value < 0).any() or (value != np.floor(value)).any():
        raise ValueError("Confusion matrices must contain nonnegative integer counts")
    return value.astype(np.int64)


def load_confusions(path):
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            if payload.get("class_names", list(NAMES)) != list(NAMES):
                raise ValueError("Class order does not match the six-class benchmark")
            payload = payload["scene_confusions"]
        else:
            payload = [payload]
        return [validated_confusion(item) for item in payload]
    return [validated_confusion(np.loadtxt(path, delimiter=","))]


def load_pairs(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Prediction/target manifest is empty")
    for row in rows:
        pred = np.load(path.parent / row["prediction"], allow_pickle=False).reshape(-1)
        target = np.load(path.parent / row["target"], allow_pickle=False).reshape(-1)
        if pred.shape != target.shape or not pred.size:
            raise ValueError("Prediction and target must have equal, nonempty lengths")
        if pred.dtype.kind not in "iu" or target.dtype.kind not in "iu":
            raise ValueError("Predictions and targets must be integer label arrays")
        if not ((pred >= 0) & (pred < 6)).all():
            raise ValueError("Predictions must use benchmark IDs 0..5")
        if not (((target >= 0) & (target < 6)) | (target == -1)).all():
            raise ValueError("Targets must use benchmark IDs 0..5 or ignore ID -1")
        yield pred.astype(np.int64), target.astype(np.int64)


def evaluate(args):
    intersections, unions, targets, confusions = [], [], [], []
    if args.pairs:
        for pred, target in load_pairs(args.pairs):
            if args.implementation == "pointcept":
                intersection, union, counts = intersection_and_union(pred, target, 6, -1)
                intersections.append(intersection)
                unions.append(union)
                targets.append(counts)
            else:
                confusions.append(confusion_from_arrays(pred, target))
    else:
        for path in args.confusion:
            confusions.extend(load_confusions(path))
    if args.implementation == "pointnet2":
        if not confusions:
            raise ValueError("No scene counts supplied")
        total = np.sum(confusions, axis=0)
        result = compute_metrics(total)
        scene_count, point_count = len(confusions), int(total.sum())
    else:
        for confusion in confusions:
            intersection = np.diag(confusion)
            target = confusion.sum(axis=1)
            prediction = confusion.sum(axis=0)
            intersections.append(intersection)
            unions.append(target + prediction - intersection)
            targets.append(target)
        if not targets:
            raise ValueError("No scene counts supplied")
        intersection = np.sum(intersections, axis=0)
        union = np.sum(unions, axis=0)
        target = np.sum(targets, axis=0)
        result = pointcept_metrics(intersection, union, target)
        scene_count, point_count = len(targets), int(target.sum())
    iou, acc, miou, macc, oa = result
    return dict(
        implementation=args.implementation, unit="fraction", class_names=list(NAMES),
        scenes=scene_count, evaluated_points=point_count, class_iou=iou.tolist(),
        class_accuracy=acc.tolist(), mIoU=miou, mAcc=macc, OA=oa,
        aggregation="Sum scene counts within one domain before computing metrics",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("pointcept", "pointnet2"), required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--confusion", nargs="+", type=Path,
                        help="Headerless 6x6 CSV(s), or JSON with class_names and scene_confusions")
    inputs.add_argument("--pairs", type=Path,
                        help="CSV columns prediction,target; .npy label paths relative to this CSV")
    parser.add_argument("--output", type=Path, help="New metrics JSON; omit to write to stdout")
    args = parser.parse_args()
    result = json.dumps(evaluate(args), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
