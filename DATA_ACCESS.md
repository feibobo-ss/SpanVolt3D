# Data access and licensing

## Native400 version 1.0

SpanVolt3D Native400 v1.0 comprises all 400 annotated transmission-line spans
across 110, 220, 500 and 1000 kV, with 5,879,971,739 native points and 13 native
annotation codes. The 400 native scenes are distributed as losslessly
compressed LAZ files in independently extractable ZIP archives. The release
retains local metric geometry and withholds absolute geographic positions.
Scene metadata link release identifiers to historical experiment scene
identifiers, anonymous project groups, voltage levels and fixed split roles.

The data are deposited in Zenodo for peer review. The reserved DOI is
https://doi.org/10.5281/zenodo.24503872. While the record is a draft, this DOI
may not resolve; anonymous reviewer access is provided with the manuscript.
The private review token is not published here. Once published, the DOI
provides public access to the dataset record and files.

## Data licence

The Native400 v1.0 point clouds and semantic labels are licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/.

Reuse and redistribution, including commercial use, are permitted under that
licence with appropriate attribution, a link to the licence and an indication
of changes. Use the dataset creators and citation recorded in Zenodo and cite
the version used. This summary does not replace the licence text.

The former project email-approval, non-commercial and non-redistribution
arrangement does not apply to the Native400 v1.0 CC BY 4.0 release. No
additional email permission is required for uses allowed by that licence.

## Representations and scope

The release contains the complete 400 selected native annotated scenes. The
original continuous engineering surveys and associated project documents
are outside its scope. The derived six-class NPY arrays used for the reference
experiments are not distributed as a separate point-cloud product.

The repository provides the label mapping, scene assignments, effective
per-scene sampling seeds, historical benchmark hashes, preparation functions,
evaluation code and archived segmentation scores. The numerical seeds retain
the documented historical-path inference and verification limits in
[reproducibility.md](docs/reproducibility.md). The published full-corpus builder
and verifier require all 400 manifest scenes. Native subsets may be reused
under CC BY 4.0 but do not by themselves reproduce the complete fixed-split
benchmark.

The reader requires uncompressed LAS. Decompress LAZ losslessly and use the
release metadata to prepare the historical scene basenames without altering
point order or coordinates. See [data details](docs/data_details.md).

## Software and other materials

Original project software uses the separate [MIT License](LICENSE), and
third-party software retains its original notices and licences; see
[LICENSING.md](LICENSING.md). The data licence does not replace those software
terms. Pretrained weights, per-point predictions and the Section 4.4 clearance
application code are not part of the current public package.
