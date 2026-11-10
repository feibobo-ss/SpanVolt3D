# Dataset and benchmark protocol

This repository describes `las_vegetation6_final_v1` with the archived XYZ-only
seven-model, three-seed experiment. The protocol is defined by the fixed
lists, saved configurations, source code and run records. Older RGB,
`others = vegetation + building`, and Ground7 experiments are different tasks.

This document and the full public manifest describe all 400 historical study
scenes. Current data access covers only approved project/scene subsets of
native labelled LAS for non-commercial research by universities and research
institutions, without republication of the supplied data. It does not include
the full corpus or frozen six-class arrays. See [data access](../DATA_ACCESS.md).
An approved subset and its delivery list do not replace the paper's fixed splits.

## Two representations

The **native labelled span data** use 13 project-specific annotation codes
(0 through 12). They include auxiliary states, and do not constitute a reported
13-class semantic-segmentation benchmark. These codes must not be interpreted
using a generic LAS/ASPRS classification dictionary.

The **six-class benchmark representation** is derived from those spans. It has
already been filtered, mapped, voxel-sampled and capped. Its stored array format
is `N x 7 float32`: `x, y, z, r, g, b, benchmark_label`. Storage of RGB does not
mean the models use RGB: all selected experiments use only XYZ as features.
Intensity, RGB, estimated normals and an additional height channel are not
model inputs. Model-internal relative coordinates and position encodings remain.

| Benchmark ID | Class | Native codes |
| --- | --- | --- |
| 0 | tower | 7 |
| 1 | conductor | 5 |
| 2 | jumper conductor | 10 |
| 3 | insulator | 8, 9 |
| 4 | shield wire | 6 |
| 5 | vegetation | 3 |

Native `0` is **Created**, whereas benchmark `0` is **tower**. Always identify
the representation before interpreting an integer label. Native 0, 1, 2, 4, 11
and 12 (Created, Unclassified, ground, building, road and water) are physically
removed in every subset, including test. The loss/evaluation ignore ID is `-1`;
it is not a seventh benchmark class. The benchmark files contain only IDs 0–5.
Suspension and horizontal insulators merge; conductors, jumpers and shield wires
remain separate. Machine-readable definitions are in `metadata/`.

## Fixed scenes and project groups

| Split in public manifest | Stored directory | Scenes | Project groups |
| --- | --- | ---: | ---: |
| train | train | 253 | 4 |
| val | val | 68 | 2 |
| test_id | test | 62 | 2 |
| test_ood_110 | ood_110 | 11 | 3 |
| test_ood_1000 | ood_1000 | 6 | 2 |

There are 400 scenes and 13 project groups. Voltage counts are 11 / 107 / 276 / 6
for 110 / 220 / 500 / 1000 kV. Development and ID evaluation use 220/500 kV;
all 110/1000-kV scenes are held out. Public scene and group IDs are strings.
Preserve their leading zeros. `splits/scene_manifest.csv` is the public source
of split truth; no random regrouping is used.

The source Excel inventory has 402 scene records, including two 220-kV entries
without corresponding LAS or frozen benchmark arrays. The historical fixed
lists already contained 400 scenes. Packaging did not delete records to achieve
the paper counts: both unmatched Excel records remain documented privately.
The original workbook and real project identifiers are not public artifacts.
See [data details](data_details.md) for the verified derivation and file checks.

Scene uniqueness and project-group disjointness are checked against the frozen
lists. They do not establish complete geographic non-overlap of all acquisition
strips or unknown project aliases; no such coordinate-level audit is claimed.

## Offline conversion: native spans to benchmark arrays

1. Read the source LAS classification field and coordinates using the original
   reader, LAS scale/offset values, and scale factor 1.0 (metres). Keep the
   original float32 conversion and RGB storage behavior.
2. Filter excluded native codes and map retained codes to 0–5.
3. Compute `floor(XYZ / 0.30)` and retain the first input-order point in each
   occupied voxel. Sort the selected indices back into input order. This is not
   centroid averaging or label voting.
4. Only when more than 1,000,000 points remain, select 1,000,000 uniformly without
   replacement using NumPy `default_rng`, then sort selected indices. Smaller
   scenes receive no additional cap sampling or padding.

The historical cap seed combines base seed 42 with a BLAKE2b hash of the
historical source identity. Simply changing a filesystem root and hashing that
new path would change selected points. The release records path-free per-scene
numeric seeds derived from the historical mount identity indicated by saved
metadata. That identity is an inference: a direct original build command was
not located. Two real scenes, one with cap sampling and one without it, were
rebuilt and matched the frozen arrays in values, point order and file SHA256.
This does not verify all 400 scenes or establish every historical seed. The
converter checks its output against each recorded benchmark hash and stops
on a mismatch. The frozen six-class arrays define the exact historical inputs;
this technical description is not an offer to supply them under the current
[data access arrangement](../DATA_ACCESS.md). See
[data details](data_details.md) for evidence and verification limits. Training
seeds 42/3407/9173 are separate from this preprocessing rule.

No additional rotation, scale normalization, geolocation-removal transform or
coordinate-system redesign was introduced. Model-side `CenterShift` centers XY
using the bounding box and shifts the minimum Z to zero; each crop/chunk is
centered in XY again. These transforms preserve metre scale. Anonymizing file
identifiers is not a claim that source geometry has been geographically scrubbed.

**Do not run native filtering or 0.30-m voxel sampling on historical benchmark
arrays.** The six-class arrays are already the input representation for training
and evaluation. Evaluation covers retained benchmark points, not every point in
the original unfiltered corridor data.

The published `build_benchmark.py` and `verify_benchmark.py` require all 400
manifest scenes and the corresponding files. They do not accept a shortened
manifest or an approved LAS subset. Received native subsets can be used directly
with their native label definitions for the approved research purpose; complete
benchmark reproduction requires the full historical data and protocol.

## Model-side sampling, budget and checkpoint selection

Training and validation use a separate 0.08-m grid. Training uses the archived
geometric augmentation sequence and random-centre crops of at most 60,000 points.
Validation still uses random within-grid sampling and one crop centred on the
middle array element; it is not deterministic full-scene validation.

Test uses identity-only processing and chunks of at most 60,000 points covering
all preprocessed scene points. Its `GridCoord` computes 0.08-m grid indices;
it does not repeat offline voxel sampling. No external test-time augmentation
or OOD adaptation is used. Some internal operations remain random, including
PointNet++ FPS and PTv3/LitePT order shuffling.

Each model uses three seeds: 42, 3407 and 9173. The budget is 200 data passes,
with validation every 20 passes. Pointcept uses ten outer epochs with a 20-fold
dataset loop; PointNet++ uses 200 epochs with loop one. These are equal data
access budgets, not a claim of identical optimizer update counts or identical
random crops. Model-specific optimizers, schedulers, precision and architectures
are retained in the saved configurations.

For each model/seed, the highest validation mIoU on the 220/500-kV validation
split selects one checkpoint, fixed for ID, OOD-110 and OOD-1000. Pointcept
retains the first tied maximum (`>`); PointNet++ accepts a later tie (`>=`).
No target-domain checkpoint selection or retraining is part of the protocol.

## Domain metrics and three-seed statistics

Accumulate point-level confusion counts across **all scenes in a domain** before
calculating metrics. For a confusion matrix with GT rows and prediction columns,
let `TP` be its diagonal, `T` its row sums, `P` its column sums, and
`U = T + P - TP`.

- Class IoU: `TP / U` with the implementation's numerical denominator guard.
- mIoU: the mean of all six class IoUs.
- Class accuracy/recall: `TP / T`; mAcc averages all six classes.
- OA: total correct points divided by total valid target points.

An absent class with zero union contributes zero, and remains in the six-class
mean. Pointcept adds `1e-10` to denominators; PointNet++ uses
`maximum(denominator, 1e-10)` for class IoU/accuracy and
`max(total_target, 1.0)` for OA. Their archived implementations are retained.
Do not average per-scene IoUs to obtain a domain score. Pointcept's instantaneous
per-scene log uses a different absent-class convention from its final domain
result, so it is not a substitute for the final result.

The reported mean and **sample** standard deviation use three seeds (`ddof=1`).
Pointcept source logs store fractions rounded to four decimals before the
historical collector converts them to percentages. PointNet++ retains the
original JSON floating-point precision. Exported tables preserve that distinction.
See [evaluation details](evaluation_details.md) for executable entries, units
and the checks against existing records.
