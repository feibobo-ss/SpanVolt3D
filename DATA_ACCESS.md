# Data access and use conditions

These conditions describe the current SpanVolt3D data-request arrangement.
They apply to point clouds and semantic labels supplied after an approved
request. The separate software licensing scope is described in
[LICENSING.md](LICENSING.md).

## Available material and scope

Eligible universities and research institutions may request the complete
400-scene collection of already segmented, labelled, span-level LAS files
retaining the 13 native annotation codes (0–12), or selected scene/project
subsets, for approved non-commercial research. The approved scene/file list
and delivery arrangements are confirmed for each request by email.

The complete frozen six-class NPY benchmark is not included in the current
access offer or the public repository. Original unsegmented engineering point
clouds and line-project materials are not provided. The anonymous 400-scene
manifest documents the historical benchmark protocol; it is not a public
download service or an automatic entitlement to receive data.

## Eligible research and permitted use

Approved access is provided **free of charge to universities and research
institutions for non-commercial research**. An application and its approval
identify the requesting research team and purpose. Access is limited to that
approved team and purpose. A different team should submit its own request.

Commercial use is not permitted under this arrangement. This includes using
the supplied data to develop, evaluate, train or improve commercial products
or services, or deploying models trained with the supplied data in commercial
products or services. The current arrangement does not provide a commercial
licensing option.

## Redistribution and research outputs

Recipients must not redistribute, publicly upload, mirror, sell or include
the supplied point clouds or labels in another downloadable dataset. This
also applies to subsets, format conversions and resampled versions that
contain supplied point coordinates or labels. Sharing within the approved
research team for the approved purpose is permitted.

Non-commercial research publications may report experimental findings,
aggregate metrics and illustrative point-cloud renderings with appropriate
attribution. Such publication must not distribute the underlying downloadable
point coordinates or labels. Report the approved scene scope and distinguish
subset experiments from results on the complete 400-scene benchmark.
Research outputs should acknowledge SpanVolt3D and cite the accompanying
paper when a verified citation becomes available.

## How to request data

Email **[feibobo721@outlook.com](mailto:feibobo721@outlook.com)** and include:

1. Your name and institutional affiliation.
2. Your research team or project contact.
3. A brief description of the non-commercial research purpose.
4. The requested scope: the complete 400-scene labelled native LAS collection
   or selected voltage/scene/project subsets, and why it is needed.
   Exact public scene IDs are not required if unknown.
5. Confirmation that the team has read and accepts these conditions.

A [plain-text request template](docs/data_request_template.txt) is provided.
Approval and the scene scope are confirmed by email. Submitting a request is
not automatic approval. The delivery method is arranged after approval; no
fixed reply deadline or public download service is promised.

## What accompanies an approved delivery

Each delivery should include the approved anonymous scene/file list, the
native label definitions, and a short note identifying the native LAS
representation and delivery version/date. The scene list is specific to that
delivery and includes all scenes approved for that request, whether the complete
400-scene collection or a selected subset. The historical manifest in this
repository does not itself authorize delivery.

The current full-benchmark command-line tools require all 400 historical
scenes. A delivered project subset can support research within its approved
scope, but it is not sufficient by itself to reproduce the paper's complete
training, ID and OOD results. See [data details](docs/data_details.md).

## Code and other materials

These data conditions do not replace licenses for the repository's software.
No pretrained weights or per-point predictions are supplied in the public
code package, and Section 4.4 application code is outside that package.
The complete frozen six-class NPY benchmark, pretrained model weights,
per-point predictions, and Section IV-D (formerly Section 4.4) application
code are outside the current delivery commitment.
