# Data representations and fixed manifest

The source annotation uses **13 project-specific native codes**, including the auxiliary states Created and Unclassified. These numbers must not be interpreted using a generic LAS/ASPRS classification table. The paper evaluates the derived **six-class** representation `las_vegetation6_final_v1`.

The full paper inventory contains two separate, corresponding layers:

| Layer | Existing files | Labels | Preparation state |
|---|---:|---|---|
| Segmented native annotation | 400 LAS files | Native codes 0–12 in LAS `classification` | Annotated scene/span files, before benchmark filtering and sampling |
| Frozen paper benchmark | 400 NPY arrays | Benchmark IDs 0–5 in column 6 | Filtered, mapped, voxel sampled and capped |

These inventory counts describe the complete study, not the data supplied with an approved request. **Current access is limited to approved project/scene subsets of the segmented native LAS layer**, retaining the 13 original annotation codes, for non-commercial research by universities and research institutions. Republishing the supplied point clouds or labels is not permitted. The current offer does not include the complete 400-scene corpus or the frozen six-class NPY layer. See [data access and use conditions](../DATA_ACCESS.md). The complete original engineering point clouds are outside this access scope.

Both historical layers are identified in the author's private inventory. No point clouds, Excel workbooks or data archives are included in this repository. The full `splits/scene_manifest.csv` documents the paper's fixed scene memberships; each approved delivery must identify its actual subset separately. The `native_filename` and `benchmark_filename` fields define anonymous naming conventions and are not download URLs or promises of delivery. A previously recorded private, single-scene check confirmed that anonymous LAS and frozen NPY copies preserve the original bytes, labels, point counts and frozen NPY SHA256. This check is evidence about those files, not a current offer of both layers or a complete anonymous data package.

The native inventory contains 5,879,971,739 points according to existing complete statistics, cross-checked here against all current LAS headers. The frozen NPY inventory contains 328,310,219 points. This release checked all NPY shapes, dtypes, file sizes and point counts against saved records, and sampled labels from one scene per voltage. It did not recount all native labels or rehash all point clouds.

## Labels and array schema

See [`native_labels.csv`](../metadata/native_labels.csv), [`benchmark_labels.csv`](../metadata/benchmark_labels.csv), and [`native_to_benchmark.csv`](../metadata/native_to_benchmark.csv).

| Native IDs | Benchmark ID | Benchmark class |
|---|---:|---|
| 7 | 0 | tower |
| 5 | 1 | conductor |
| 10 | 2 | jumper conductor |
| 8, 9 | 3 | insulator |
| 6 | 4 | shield wire |
| 3 | 5 | vegetation |

Native 0, 1, 2, 4, 11 and 12 (Created, Unclassified, ground, building, road surface and water) are physically removed in **every** subset before voxel sampling. `ignore_id=-1` is the model/evaluation sentinel, not a stored seventh class. Native 0 is Created; benchmark 0 is tower. The two meanings must not be mixed.

Frozen arrays are `float32`, shape `N × 7`, columns `X, Y, Z, R, G, B, benchmark_label`. RGB columns are retained to preserve the historical file representation. The current benchmark models consume XYZ only. Intensity is not an input feature. Do not remap benchmark labels or repeat offline sampling on historical six-class NPY files.

## How the manifest was derived

The four original voltage-specific Excel workbooks are scene inventories, **not spreadsheets containing train/val/test assignments**. Each has one saved worksheet and no formula cells. Column A holds the source-file/project description and is carried forward across blank rows; B contains the scene number. Other columns describe the scene or tower type, with an additional RGB column in one workbook. Original values, sheet names, cell addresses, row contents and workbook hashes remain in the private audit records.

The preserved chain is:

1. Match Excel voltage and saved scene number to existing LAS/NPY names, including original suffixes.
2. Retain the original project/corridor families from the existing fixed split implementation. These are deliberately coarser than individual Excel rows.
3. Retain membership from the existing `las_final_v1` lists and the unchanged lists used by `las_vegetation6_final_v1`.
4. Check those lists against all current native and benchmark filenames and the historical run inventory.
5. Apply the existing global `SCENE_0001`–`SCENE_0400` and `PROJECT_01`–`PROJECT_13` anonymous mapping, preserving strings and leading zeroes.

No grouping or random split was recomputed. Each project occurs in one subset. Anonymous filename sorting preserves the original sorted file order within each runtime subset. All 400 manifest scenes have an Excel match and both a current native file and a frozen six-class file.

The Excel inventories have **402 rows**, including two 220-kV entries that were already absent from the original 400-scene frozen corpus. These two rows remain recorded privately; they were not deleted from the workbooks or added to the public benchmark. This is a documented inventory discrepancy, not a newly modified split.

| Public `split` | Runtime `data_split` directory | Scenes | Project groups |
|---|---|---:|---:|
| train | train | 253 | 4 |
| val | val | 68 | 2 |
| test_id | test | 62 | 2 |
| test_ood_110 | ood_110 | 11 | 3 |
| test_ood_1000 | ood_1000 | 6 | 2 |

The voltage counts are 11 / 107 / 276 / 6 for 110 / 220 / 500 / 1000 kV. `points` means the recorded six-class point count; `native_points` is the LAS header count. `benchmark_sha256` is copied from the historical run inventory; it was not recomputed for all arrays during this release.

## Historical full-corpus preprocessing

The numerical functions in [`preprocessing/core.py`](../preprocessing/core.py) are unchanged extracts of the actual processing implementation. The public wrapper preserves the selected builder's sequence:

1. Decode LAS XYZ using its stored scale and offsets, cast to `float32`, retain scale 1.0 and the existing RGB handling.
2. Filter native labels and apply the six-class mapping.
3. Compute `floor(XYZ / 0.30)`. Keep the first point in file order in each voxel, then sort retained indices to preserve file order.
4. Only when the result exceeds 1,000,000 points, choose 1,000,000 indices uniformly without replacement with NumPy `default_rng`, then sort indices to retain original order.
5. Save the same seven-column representation.

The selected builder does not introduce a further translation, rotation, normalization or scale change. The already segmented native LAS is the input layer; earlier engineering-to-local-coordinate preparation is outside this published conversion. Model-side coordinate centering, the 0.08 m grid and 60,000-point crops/chunks belong to a separate stage.

The historical cap seed is `(42 + integer(blake2b(source_identity, digest_size=8))) mod 2^32`, with the digest interpreted little-endian. It is not simply seed 42. The public manifest stores the resulting integer so that private path strings are unnecessary. **The historical root used for these integers is inferred from saved build/run metadata and the builder's relative path rule. A direct original build invocation was not located.** `sampling_seed_basis` remains unchanged. Synthetic tests confirm equivalent seed handling. Two real native scenes were additionally converted with the public `process_scene` and their existing manifest seeds:

| Scene | Native points | After filtering/mapping | After 0.30 m voxel sampling | After cap | Comparison with frozen NPY |
|---|---:|---:|---:|---:|---|
| SCENE_0120 | 1,584,396 | 1,522,895 | 626,759 | 626,759; cap not triggered | Same float32 array values, point order and file SHA256 |
| SCENE_0122 | 2,345,363 | 2,260,181 | 1,033,892 | 1,000,000; cap triggered | Same float32 array values, point order and file SHA256 |

These checks validate the two selected scenes, including one actual random-cap case. They do not validate all 400 reconstructions or replace the missing original build invocation. The historical frozen NPY layer defines the exact paper input bytes; it is outside the current data offer. The builder checks each reconstructed NPY against the recorded SHA256 and stops on a difference.

The two entrypoints below require the complete 400-scene manifest and corresponding historical data. **They do not accept an approved LAS subset or a shortened manifest.** Keep the full public manifest unchanged. NumPy and Python's standard library are sufficient for these entrypoints, which are retained to document the complete historical preparation and verification workflow:

```bash
# Complete historical frozen six-class corpus only:
python3 preprocessing/verify_benchmark.py --data-root /path/to/benchmark6
# Optional byte verification of that complete corpus:
python3 preprocessing/verify_benchmark.py --data-root /path/to/benchmark6 --sha256

# Complete historical segmented NATIVE LAS corpus only; no outputs are written:
python3 preprocessing/build_benchmark.py \
  --native-root /path/to/native \
  --output-root /path/to/benchmark6 \
  --dry-run
# Remove --dry-run to process. Use a new, separate output directory.
```

The native dry run verifies fixed manifest counts, paths, point format, LAS header point counts and file lengths. The six-class default verification checks fixed membership, array shape/dtype and point count without scanning every point. The optional `--sha256` compares all file bytes. No command infers the representation from label value zero alone.

Recipients of an approved LAS subset can use its native labels directly for the approved research purpose, following `metadata/native_labels.csv` and the scene list supplied with that delivery. A project/scene subset is not the complete six-class benchmark and does not by itself reproduce the paper's full fixed-split results. No subset conversion or subset verification command is provided by the two full-corpus entrypoints above.

Release checks passed for source-function equivalence, a small synthetic LAS read/filter/voxel comparison, cap-boundary and effective-seed equivalence, both `--help` entrypoints, a 400-header native dry run, and 400 benchmark header checks. The subsequent limited checks verified all 400 private-to-public manifest associations, one anonymous delivery pair and the two real conversions listed above. The anonymous delivery check reused the same header, array and hash checks in a private script; it did not lower the public verifier's default 400-scene requirement. Original files and approved manifests were unchanged. No training, inference, corpus-wide reconstruction or corpus-wide point-cloud rehash was performed.
