"""Collect historical XYZ scores and summarize exactly three fixed seeds.

Collection follows the original seven-model collector. The output directory
must be new, so historical paper records cannot be replaced by this command.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

from historical_readers import pn_metrics, read_metrics
from protocol import DOMAINS, METRICS, MODELS, SEEDS


def collect(run_root):
    rows = []
    for model in MODELS:
        reader = pn_metrics if model == "pointnet2" else read_metrics
        for seed in SEEDS:
            for domain in DOMAINS:
                metrics = reader(run_root / model / f"seed{seed}", domain)
                if metrics is None:
                    raise ValueError(f"Missing completed evaluation: {model}/seed{seed}/{domain}")
                rows.append(dict(model=model, seed=seed, domain=domain,
                                 status="evaluated", unit="percent", **metrics))
    return rows


def read_scores(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(rows):
    expected = {(m, s, d) for m in MODELS for s in SEEDS for d in DOMAINS}
    seen = set()
    normalized = []
    for row in rows:
        key = row["model"], int(row["seed"]), row["domain"]
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate model/seed/domain: {key}")
        if row["unit"] != "percent" or row["status"] != "evaluated":
            raise ValueError("Each input score must be evaluated and have unit=percent")
        seen.add(key)
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in values.values()):
            raise ValueError(f"Scores outside [0,100]: {key}")
        normalized.append(dict(model=key[0], seed=key[1], domain=key[2],
                               status="evaluated", unit="percent", **values))
    if seen != expected:
        raise ValueError(f"Missing model/seed/domain records: {sorted(expected - seen)}")
    summary = []
    for model in MODELS:
        for domain in DOMAINS:
            group = [r for r in normalized if r["model"] == model and r["domain"] == domain]
            result = dict(model=model, domain=domain, n_seeds=3, std_ddof=1,
                          status="complete", unit="percent")
            for metric in METRICS:
                xyz = [r[metric] for r in group]
                result[f"{metric}_mean"] = statistics.mean(xyz)
                result[f"{metric}_std"] = statistics.stdev(xyz)
            summary.append(result)
    return normalized, summary


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-root", type=Path, help="Existing seven-model run tree")
    inputs.add_argument("--scores", type=Path, help="Existing XYZ per-seed CSV, unit=percent")
    parser.add_argument("--output-dir", type=Path, required=True, help="New destination directory")
    args = parser.parse_args()
    rows = collect(args.run_root) if args.run_root else read_scores(args.scores)
    rows, summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "xyz_seed_results.csv", rows)
    write_csv(args.output_dir / "xyz_three_seed_summary.csv", summary)
    print("Collected 21 runs / 63 domain records; 21 summaries with sample SD (ddof=1).")


if __name__ == "__main__":
    main()
