"""Original historical-result readers; inputs are read only.

The published subset collects XYZ scores and omits the older RGB comparison.
Pointcept logs use four decimal places; PointNet++ JSON keeps full precision.
"""
import json
import math
import re
from protocol import NAMES, METRICS

def read_metrics(run, domain):
    path = run / ("train.log" if domain == "id" else f"eval_{domain}/eval.txt")
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    if domain == "id":
        if "Start Precise Evaluation" not in text:
            return None
        text = text.rsplit("Start Precise Evaluation", 1)[1]
        if "Testing on model_best" not in text:
            return None
    matches = list(re.finditer(r"Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)", text))
    if not matches:
        return None
    match = matches[-1]
    tail = text[match.end():]
    if "End Evaluation" not in tail:
        return None
    classes = re.findall(r"Class_(\d+) - (\w+) Result: iou/accuracy ([\d.]+)/([\d.]+)", tail)
    if [(int(i), n) for i, n, _, _ in classes] != list(enumerate(NAMES)):
        return None
    scores = dict(zip(METRICS[:3], map(float, match.groups())))
    scores.update({name: float(iou) for _, name, iou, _ in classes})
    return {key: value * 100 for key, value in scores.items()}

def pn_metrics(run, domain):
    split = "test" if domain == "id" else domain
    path = run / f"test_{split}/summary.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None  # Evaluation may still be writing its summary.
    assert result["class_names"] == list(NAMES) and result["split"] == split, path
    assert result["total_points"] == {"test": 44437090, "ood_110": 8024304, "ood_1000": 4528547}[split], path
    assert len(result["class_iou"]) == 6, path
    values = {k: result[k] for k in METRICS[:3]}
    values.update(zip(NAMES, result["class_iou"]))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()), path
    return {k: v * 100 for k, v in values.items()}
