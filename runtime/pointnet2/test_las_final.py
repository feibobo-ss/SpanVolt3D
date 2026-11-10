import argparse
import csv
import importlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from las_final_data import build_las_final_test_dataset, load_class_names, feature_channels_from_config


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "models"))


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ LAS final benchmark evaluator")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model", default="pointnet2_sem_seg_msg_las_final")
    parser.add_argument("--in-channels", type=int, choices=(3, 6), default=6)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("test", "ood_110", "ood_1000"), required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save one full-resolution prediction array per scene.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_from_arrays(pred, target, num_classes=6):
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_metrics(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    target = confusion.sum(1).astype(np.float64)
    prediction = confusion.sum(0).astype(np.float64)
    union = target + prediction - intersection
    iou = intersection / np.maximum(union, 1e-10)
    acc = intersection / np.maximum(target, 1e-10)
    miou = float(iou.mean())
    macc = float(acc.mean())
    oa = float(intersection.sum() / max(target.sum(), 1.0))
    return iou, acc, miou, macc, oa


def build_logger(save_path):
    logger = logging.getLogger(f"pointnet2-las-test-{save_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(save_path) / "test.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def main():
    args = parse_args()
    class_names = load_class_names(args.config_file)
    seed_everything(args.seed)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = build_logger(save_path)
    dataset, cfg = build_las_final_test_dataset(args.config_file, args.split)
    if feature_channels_from_config(cfg, "test") != args.in_channels:
        raise ValueError("Model channels do not match test features")

    module = importlib.import_module(args.model)
    device = torch.device("cuda")
    model = (module.get_model(6) if args.in_channels == 6
             else module.get_model(6, in_channels=args.in_channels)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    confusion = np.zeros((6, 6), dtype=np.int64)
    scene_rows = []
    scene_confusion_dir = save_path / "scene_confusions"
    scene_confusion_dir.mkdir(exist_ok=True)
    prediction_dir = save_path / "result"
    if args.save_predictions:
        prediction_dir.mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    logger.info(
        "Protocol: split=%s scenes=%d features=%s point_max=60000 grid=0.08 TTA=off checkpoint_epoch=%s seed=%d",
        args.split,
        len(dataset),
        "XYZ(3)" if args.in_channels == 3 else "XYZ+RGB(6)",
        checkpoint.get("epoch", "unknown"),
        args.seed,
    )
    with torch.no_grad():
        for scene_idx in range(len(dataset)):
            start = time.time()
            sample = dataset[scene_idx]
            fragments = sample["fragment_list"]
            target = np.asarray(sample["segment"], dtype=np.int64).reshape(-1)
            score = torch.zeros((target.size, 6), dtype=torch.float32, device=device)
            votes = torch.zeros(target.size, dtype=torch.int32, device=device)

            for fragment in fragments:
                feat = fragment["feat"].float().to(device, non_blocking=True)
                index = fragment["index"].long().to(device, non_blocking=True)
                logits = model(feat.transpose(0, 1).unsqueeze(0).contiguous())[0]
                probability = torch.softmax(logits, dim=-1)
                score.index_add_(0, index, probability)
                votes.index_add_(0, index, torch.ones_like(index, dtype=torch.int32))

            if torch.any(votes == 0):
                missing = int((votes == 0).sum().item())
                raise RuntimeError(f"Scene {sample['name']} has {missing} points without an inference vote")
            pred = score.argmax(1).cpu().numpy()
            scene_confusion = confusion_from_arrays(pred, target)
            confusion += scene_confusion
            scene_iou, scene_acc, scene_miou, scene_macc, scene_oa = compute_metrics(
                scene_confusion
            )
            elapsed = time.time() - start
            scene_rows.append(
                {
                    "scene": sample["name"],
                    "points": int(target.size),
                    "fragments": int(len(fragments)),
                    "seconds": elapsed,
                    "points_per_second": float(target.size / max(elapsed, 1e-9)),
                    "mIoU": scene_miou,
                    "mAcc": scene_macc,
                    "OA": scene_oa,
                    **{
                        f"iou_{name}": float(value)
                        for name, value in zip(class_names, scene_iou)
                    },
                }
            )
            np.save(scene_confusion_dir / f"{sample['name']}.npy", scene_confusion)
            if args.save_predictions:
                np.save(prediction_dir / f"{sample['name']}_pred.npy", pred.astype(np.uint8))
            logger.info(
                "Test: [%d/%d] %s points=%d fragments=%d time=%.1fs",
                scene_idx + 1,
                len(dataset),
                sample["name"],
                target.size,
                len(fragments),
                elapsed,
            )

    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_start

    iou, acc, miou, macc, oa = compute_metrics(confusion)
    logger.info("Val result: mIoU/mAcc/allAcc %.4f/%.4f/%.4f", miou, macc, oa)
    for idx, name in enumerate(class_names):
        logger.info("Class_%d - %s Result: iou/accuracy %.4f/%.4f", idx, name, iou[idx], acc[idx])

    summary = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mIoU": miou,
        "mAcc": macc,
        "OA": oa,
        "class_names": class_names,
        "class_iou": iou.tolist(),
        "class_accuracy": acc.tolist(),
        "parameter_count": parameter_count,
        "total_points": int(sum(row["points"] for row in scene_rows)),
        "total_seconds": total_seconds,
        "points_per_second": float(
            sum(row["points"] for row in scene_rows) / max(total_seconds, 1e-9)
        ),
        "seconds_per_scene": float(total_seconds / max(len(scene_rows), 1)),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    with open(save_path / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    np.savetxt(save_path / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    with open(save_path / "scene_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)


if __name__ == "__main__":
    main()
