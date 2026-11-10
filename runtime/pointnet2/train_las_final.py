import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from las_final_data import LASFinalTrainValDataset, load_class_names


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "models"))


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ LAS final benchmark trainer")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model", default="pointnet2_sem_seg_msg_las_final")
    parser.add_argument("--in-channels", type=int, choices=(3, 6), default=6)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval-step", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_from_batch(pred, target, num_classes=6):
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    target = confusion.sum(1).astype(np.float64)
    prediction = confusion.sum(0).astype(np.float64)
    union = target + prediction - intersection
    iou = intersection / np.maximum(union, 1e-10)
    acc = intersection / np.maximum(target, 1e-10)
    return iou, acc, float(iou.mean()), float(acc.mean()), float(intersection.sum() / max(target.sum(), 1.0))


def build_logger(save_path):
    logger = logging.getLogger(f"pointnet2-las-final-{save_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(save_path) / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def evaluate(model, criterion, loader, device):
    model.eval()
    confusion = np.zeros((6, 6), dtype=np.int64)
    losses = []
    with torch.no_grad():
        for feat, target in loader:
            feat = feat.to(device, non_blocking=True).transpose(1, 2).contiguous()
            target = target.to(device, non_blocking=True)
            logits = model(feat)
            loss = criterion(logits, target)
            losses.append(float(loss.detach().cpu()))
            pred = logits.argmax(-1).cpu().numpy().reshape(-1)
            truth = target.cpu().numpy().reshape(-1)
            confusion += confusion_from_batch(pred, truth)
    return float(np.mean(losses)), confusion, metrics(confusion)


def main():
    args = parse_args()
    class_names = load_class_names(args.config_file)
    seed_everything(args.seed)
    save_path = Path(args.save_path)
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger = build_logger(save_path)
    with open(save_path / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

    train_dataset = LASFinalTrainValDataset(args.config_file, "train")
    val_dataset = LASFinalTrainValDataset(args.config_file, "val")
    if train_dataset.feature_channels != args.in_channels or val_dataset.feature_channels != args.in_channels:
        raise ValueError("Model channels do not match train/validation features")
    generator = torch.Generator().manual_seed(args.seed)

    def worker_init(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    module = importlib.import_module(args.model)
    device = torch.device("cuda")
    model = (module.get_model(6) if args.in_channels == 6
             else module.get_model(6, in_channels=args.in_channels)).to(device)
    criterion = module.get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_miou = -1.0
    logger.info(
        "Protocol: train=%d val=%d features=%s point_max=60000 grid=0.08 loss=CE+Lovasz class_sampling=off epochs=%d seed=%d",
        len(train_dataset), len(val_dataset), "XYZ(3)" if args.in_channels == 3 else "XYZ+RGB(6)", args.epoch, args.seed,
    )

    for epoch in range(1, args.epoch + 1):
        start = time.time()
        lr = max(args.learning_rate * (args.lr_decay ** ((epoch - 1) // args.step_size)), 1e-5)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses = []
        for step, (feat, target) in enumerate(train_loader, 1):
            feat = feat.to(device, non_blocking=True).transpose(1, 2).contiguous()
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # PointNet++ computes pairwise squared distances directly.  FP16
            # autocast can overflow/cancel on corridor-scale XYZ coordinates,
            # so this architecture is intentionally trained in stable FP32.
            logits = model(feat)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                logger.info(
                    "Train: [%d/%d][%d/%d] loss %.4f lr %.6f elapsed %.1fs",
                    epoch, args.epoch, step, len(train_loader), losses[-1], lr, time.time() - start,
                )

        logger.info("Epoch %d/%d train_loss %.6f time %.1fs", epoch, args.epoch, np.mean(losses), time.time() - start)
        if epoch % args.eval_step == 0 or epoch == args.epoch:
            val_loss, confusion, result = evaluate(model, criterion, val_loader, device)
            iou, acc, miou, macc, oa = result
            logger.info("Val result: mIoU/mAcc/allAcc %.4f/%.4f/%.4f loss %.6f", miou, macc, oa, val_loss)
            for idx, name in enumerate(class_names):
                logger.info("Class_%d - %s Result: iou/accuracy %.4f/%.4f", idx, name, iou[idx], acc[idx])
            state = {
                "epoch": epoch,
                "seed": args.seed,
                "model": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": max(best_miou, miou),
            }
            torch.save(state, checkpoint_path / "model_last.pth")
            np.savetxt(save_path / f"val_confusion_epoch_{epoch}.csv", confusion, delimiter=",", fmt="%d")
            if miou >= best_miou:
                best_miou = miou
                torch.save(state, checkpoint_path / "model_best.pth")
                logger.info("Best validation mIoU updated to: %.4f", best_miou)

    logger.info("Training complete. Best mIoU: %.4f", best_miou)


if __name__ == "__main__":
    main()
