# Evaluation and historical result provenance

The evaluation inputs are predictions and ground-truth labels for the prepared
six-class benchmark. Native annotation codes are not accepted as benchmark IDs.
No filtering, voxel sampling, model inference, or coordinate transformation is
performed by the offline evaluation commands below. A prediction vector must
correspond point for point to its prepared scene's ground-truth label vector.

The order is `tower`, `conductor`, `jumper_conductor`, `insulator`, `shield_wire`,
`vegetation`, with IDs 0 through 5. Benchmark ignore ID is -1. In particular,
benchmark ID 0 is tower; it is not native annotation code 0.

## Count aggregation and numerical conventions

Within each evaluation domain, sum all scene counts before calculating metrics.
For a confusion matrix, rows are ground truth and columns are predictions.
Let `I` be its diagonal, `T` its row sums, `P` its column sums, and
`U = T + P - I`. IoU is class intersection over union; class accuracy is recall;
mIoU and mAcc are unweighted means across all six classes; OA is total correct
predictions divided by total evaluated points. Absent classes contribute zero
to the six-class means. Do not average scene mIoU or scene OA to form a domain
score.

Both actual framework conventions are retained in `evaluation/historical_metrics.py`:

| Implementation | Per-class IoU / accuracy denominator | OA denominator |
| --- | --- | --- |
| Pointcept models | `U + 1e-10` / `T + 1e-10` | `sum(T) + 1e-10` |
| PointNet++ | `maximum(U, 1e-10)` / `maximum(T, 1e-10)` | `max(sum(T), 1.0)` |

These are the original arithmetic rules, including their small epsilon
difference. Empty denominators yield zero. The offline command requires an
explicit implementation choice. It sums counts using the same final-domain
equations and returns metrics as fractions in [0, 1].

## Offline evaluation

Only NumPy is required. From the repository root:

```bash
python3 evaluation/evaluate.py --help
python3 evaluation/evaluate.py --implementation pointcept \
  --confusion evaluation/synthetic_counts.json
python3 evaluation/evaluate.py --implementation pointnet2 \
  --confusion evaluation/synthetic_counts.json
```

The included JSON is explicitly synthetic: two small artificial confusion
matrices for an interface check. Its scores are unrelated to the paper.

For existing count records, pass one or more headerless 6-by-6 CSV files with
integer counts. Alternatively, JSON may contain `class_names` and a
`scene_confusions` list of 6-by-6 matrices. Supply exactly one domain and one
model/seed to an evaluation call. Do not mix a domain-total matrix with its
constituent scene matrices, which would count points twice.

For saved predictions, create a CSV with columns `prediction,target`. Each row
names two integer `.npy` label arrays of equal length. Paths are relative to the
CSV directory, or absolute paths supplied by the recipient. Use benchmark
prediction IDs 0..5 and target IDs 0..5 or -1. Files are loaded with
`allow_pickle=False`.

```bash
python3 evaluation/evaluate.py --implementation pointcept \
  --pairs /path/to/recipient/domain_prediction_pairs.csv \
  --output /path/to/new/domain_metrics.json
```

`--output` must name a new file. This command does not start inference. Real
point labels, predictions, count records with scene identifiers, and weights
are excluded from this release candidate.

## Checkpoint selection and three-seed summaries

Every model/seed selects the checkpoint with the highest 220/500-kV validation
mIoU and uses that checkpoint for ID, OOD-110, and OOD-1000. The original
Pointcept checkpoint saver updates on a strict improvement (`>`); PointNet++
updates on `>=`, retaining the later checkpoint in an exact tie. No target-domain
checkpoint selection is part of this benchmark.

`results/historical_xyz_seed_results.csv` contains 63 existing XYZ evaluation
records: seven models, seeds 42/3407/9173, and three domains.
`results/historical_xyz_three_seed_summary.csv` contains their 21 saved
three-seed summaries. Both are whitelist extracts of the completed
`las_vegetation6_final_v1` XYZ-only experiment records; the older RGB comparison
columns have been omitted. Existing numeric strings were retained. These are
historical results, not newly run experiments or a claim of full reproduction.

The historical collector read the six Pointcept models from the final precise
ID evaluation in `train.log` and each OOD `eval.txt`; PointNet++ was read from
its `summary.json`. Pointcept fractions had already been rounded to four
decimals in the logs before multiplication by 100. PointNet++ values retained
the JSON's saved precision before multiplication by 100. Published result CSVs
therefore use percent; standard deviations use the same percent scale.
Per-class columns contain class IoU, not class accuracy.

Three-seed summaries use `statistics.mean` and `statistics.stdev`, i.e. sample
standard deviation with `ddof=1`. All three seeds must be present. Seeds are
aggregated after each seed's domain metric has been computed; counts are not
pooled across seeds.

```bash
python3 evaluation/aggregate.py --help
python3 evaluation/aggregate.py \
  --scores results/historical_xyz_seed_results.csv \
  --output-dir /tmp/spanvolt_summary
```

The output directory must not exist. This protects the historical result
files. The command produces `xyz_seed_results.csv` and
`xyz_three_seed_summary.csv`. It rejects duplicate, missing, nonfinite, or
out-of-range records and requires `unit=percent`.

The same command can collect an existing historical-format experiment tree:

```bash
python3 evaluation/aggregate.py --run-root /path/to/recipient/experiment \
  --output-dir /path/to/new/summary
```

The expected layout is `<model>/seed<seed>/`. For the six Pointcept models,
use `train.log`, `eval_ood_110/eval.txt`, and `eval_ood_1000/eval.txt`.
The ID record must be the post-training `PreciseEvaluator` result in `train.log`,
including its completion and `model_best` markers. A standalone Pointcept ID
test writes `eval.txt` under its own output directory; this historical reader
does not collect that file automatically or discover arbitrary output layouts.
For `pointnet2`, use `test_test/summary.json`, `test_ood_110/summary.json`, and
`test_ood_1000/summary.json`. Model identifiers are `pointnet2`, `spunet`,
`oacnns`, `ptv1`, `ptv2`, `ptv3`, and `litept`.
The separate PointNet++ test commands in [the configuration guide](../configs/README.md)
produce these three output paths.

The original readers are retained, including class-order/completion checks
and PointNet++ fixed-domain point totals (44,437,090 / 8,024,304 / 4,528,547).
This collection mode targets this fixed historical benchmark, not arbitrary
datasets. The `--scores` mode is the portable summary interface for already
collected domain scores.

## Checks performed for this candidate

Static checks covered syntax and entrypoint parameters. Synthetic runs checked
the original kernels against the extracted kernels, ignored labels, absent
classes, all-zero counts, both offline interfaces, and CSV summarization.

Existing-record checks matched all 63 XYZ records and all 21 mean/SD rows to
the saved historical tables. One original PointNet++ seed-42 OOD-1000 record
was checked by summing its six saved scene confusion matrices: their sum
matched the saved domain matrix and all saved metrics exactly, over 4,528,547
points. These count records remain private. The domain mIoU was
0.16156988131477995; the unweighted scene mIoU mean was
0.15720436160943826, confirming the distinction.

All 21 runs had checkpoint-selection evidence in existing logs. For PointNet++,
the saved validation count matrices additionally confirmed selected epochs
80/120/200 for seeds 42/3407/9173. Checkpoint tensors were not loaded. Source
hashes and detailed verification records are retained outside the public package.

No training, model inference, full test rerun, dependency upgrade, or replacement
of historical result files was performed. Later visualization and application
diagnostic scripts were not used as sources for these main-table results.
