"""Historical benchmark kernels, extracted without changing arithmetic.

Pointcept kernels retain their upstream provenance; see THIRD_PARTY_NOTICES.md.
No training framework or GPU is required for these NumPy operations.
"""
import numpy as np

def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    # Keep only points whose label is not ignore_index.
    mask = target != ignore_index
    output, target = output[mask], target[mask]

    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

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

def pointcept_metrics(intersection, union, target):
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    return iou_class, accuracy_class, float(mIoU), float(mAcc), float(allAcc)
