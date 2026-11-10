# Reproducibility and validation limits

This repository targets the archived `las_vegetation6_final_v1` XYZ-only
benchmark. The evidence below distinguishes source/configuration checks,
limited real-data conversion checks and runtime construction from full training
or benchmark reproduction.

## Selected implementation

The selected run family began on 2026-09-08, with its five-model second stage
added on 2026-09-09. It contains seven models and seeds 42/3407/9173.
All 18 Pointcept saved configurations and the three PointNet++ data/training
configurations are included with portable paths. PointNet++ retains its own
trainer and MSG implementation. The other six models retain their Pointcept
implementations; see [model details](../configs/README.md).

Fifteen unique source files covered by historical pre-run audit hashes match
the selected source. These include the shared dataset, transforms, train/test
engines, checkpoint hooks, PointNet++ code and PTv1/PTv2/OA-CNNs/LitePT backbones.
This is not a historical hash of the entire repository. PTv3/SpUNet backbones
and other necessary files have current SHA256 identities; a full historical
source snapshot or upstream commit tied to every run was not found.

The runtime contains explicitly selected files. Required package registrations
and imports were retained; unused heads/imports and path examples were removed
where needed. Selected forward, sampling, loss, optimizer, checkpoint and metric
logic is preserved. Command wrappers invoke the existing implementations.

## Limited checks on 2026-11-09

| Check | Result and boundary |
| --- | --- |
| PointNet++ test commands | Three documented commands passed dry-run; generated seed, split, checkpoint and `test_test` / `test_ood_110` / `test_ood_1000` output paths agree with the original reader |
| Historical result-reader layout | Synthetic records passed the three PointNet++ reader paths. A separate Pointcept case confirmed ID comes from post-training `PreciseEvaluator` in `train.log`, not arbitrary standalone `eval.txt` |
| Private-to-public manifest association | All 400 existing associations matched; the approved manifest, label tables, seeds and historical results were unchanged |
| Anonymous delivery | One native LAS / frozen NPY pair, SCENE_0120, was copied under the public names in a private temporary directory. Recipient-side names, native labels, benchmark labels, shape/dtype/count and file hashes passed; this is not a full 400-scene delivery validation |
| Native conversion without cap | SCENE_0120: 1,584,396 native points, 626,759 after 0.30-m sampling; cap not triggered. Released `process_scene` output matches the frozen array values, order and NPY SHA256 |
| Native conversion with cap | SCENE_0122: 2,345,363 native points, 1,033,892 after 0.30-m sampling, capped to 1,000,000. Existing manifest seed used once; released output matches frozen values, order and NPY SHA256 |
| Public runtime | Real package imports, seven seed-42 configuration parses and seven CPU model constructions passed in an existing compatible environment. Local model/engine/adapter module paths were checked against the public runtime |
| LitePT dependencies | FlashAttention 2.8.3 and PointROPE 0.0.0 imported from the existing installed runtime; LitePT construction passed with those dependencies present |
| Final source package | File contents and ZIP membership checked against the approved baseline and exclusions; private evidence records exact changed files and archive hashes |

The two reconstructions used Python 3.10.12 and NumPy 2.2.6. The capped case
actually entered random sampling; it was not classified as capped solely from
its final point count. Both saved files are byte-identical to their respective
frozen historical files, so there was no NPY-header-only discrepancy to explain.
Details of the processing stages are in [data details](data_details.md).

Runtime checks used the existing Python 3.10.20 / PyTorch 2.5.0 / CUDA 12.4
package environment with **spconv 2.3.8** and PointOps 1.0. The installed
PointOps Python sources matched the included extension sources. LitePT's
existing extra installed dependencies were checked separately. These are
**versions observed in this verification environment**, not a reconstructed
per-run historical package lock. See [environment validation](../environments/README.md).

No GPU forward pass, training, formal ID/OOD inference, full metric evaluation,
package installation, upgrade or extension compilation was performed in these
runtime checks. CPU construction verifies imports and registration, not the
execution of GPU kernels. No unlisted old project model modules were used to
make the public package import successfully.

## Previously recorded checks

The initial source preparation recorded Python syntax and configuration
comparisons, all seven model train/test dry-runs, synthetic preprocessing and
metric equivalence, and header/size checks for 400 LAS and 400 NPY files. The
four original Excel workbooks matched their saved hashes; fixed membership
contained 400 unique scenes and 13 disjoint project groups, with two Excel-only
records documented. Those broad checks were not rerun in the limited closeout.

The initial historical-result audit matched 63 XYZ domain records and 21
three-seed summaries to the saved tables, including sample SD (`ddof=1`). One
PointNet++ seed42 OOD1000 record had six saved scene confusion matrices summing
to the domain matrix over 4,528,547 points. Its domain mIoU was
`0.16156988131477995`; the mean of its scene mIoUs was
`0.15720436160943826`, confirming the domain-count aggregation distinction.
All 21 runs had existing checkpoint-selection evidence. These were record
checks, not fresh inference, and were not repeated in this closeout.

## Remaining interpretation limits

- Historical cap sampling hashed a source-path identity. The path-free numeric
  seeds in the manifest remain marked `inferred_historical_build_path` because
  a direct original build invocation was not located. The two real checks
  support the existing conversion, including one capped scene; they do not
  establish all 400 seeds or validate all 400 reconstructions. The historical
  frozen six-class arrays define the exact paper input bytes. Conversion stops
  if a generated file fails its recorded historical hash.
- The current [access arrangement](../DATA_ACCESS.md) covers only approved
  project/scene subsets of native LAS for non-commercial research by universities
  and research institutions, without republication of the supplied data. The
  full 400-scene corpus and frozen six-class arrays are outside the current
  offer. The publicly documented full protocol does not make a partial delivery
  sufficient for reproducing the complete benchmark.
- One anonymous LAS/NPY pair was validated in the recorded checks; no complete
  delivery corpus has been validated under anonymous names. This evidence does
  not extend the current access offer. The public builder and verifier retain
  their requirement for all 400 scenes and do not accept a shortened manifest
  or a delivered native subset.
- Real imports and CPU construction do not establish GPU forward correctness,
  CUDA-kernel numerical behavior or training/inference reproducibility. A fresh
  installation from the unpinned recipe, full per-run package freeze, all
  extension build options and a complete historical source revision remain
  unavailable or unverified.
- Pointcept scores retain the precision of four-decimal fractional logs;
  PointNet++ scores retain the saved JSON precision. Summaries do not add
  measurement precision. Domain metrics aggregate point counts before computing
  scores; seeds are summarized afterward.
- Source storage includes RGB; model features use XYZ only. Test inputs already
  exclude background and auxiliary native codes. The benchmark evaluates
  retained points rather than an unfiltered corridor scene.
- Project-group disjointness follows the frozen inventory and grouping. It is
  not a coordinate-level proof of non-overlap of every acquisition strip.
- Later visualization and Section 4.4 application diagnostics are not sources
  for the main segmentation table. Section 4.4 application code is not included
  in this repository.

See [data access and use conditions](../DATA_ACCESS.md) for the current offer and
[paper and contact information](../README.md#paper-and-contact) for the manuscript
and repository status, contact details and applicable code licensing.
