# Recorded runtime and installation

This candidate contains the selected model source, loaders, transforms, training and test engines, and CUDA extension source. It does not contain pretrained weights, wheels, an environment image, or a complete historical package lock. No packages were installed or upgraded during release preparation.

The recorded base recipe uses Python 3.10, PyTorch 2.5.0, torchvision 0.20.0 and CUDA 12.4. Training used an NVIDIA RTX 4090. `pointcept-recorded.yml` preserves the existing recipe with unused PEFT/transformers and local build directives removed. Unpinned packages in that recipe are not claimed to reproduce their historical versions. The implementation uses **`spconv.pytorch`, package family `spconv-cu124`**. The existing benchmark image inspected on 2026-11-09 contains **spconv 2.3.8**. That is an observed validation-environment version; it is not a complete version audit of every historical run.

On a separate compatible machine, the available recipe and source build commands are:

```bash
conda env create -f environments/pointcept-recorded.yml
conda activate pointcept
(cd runtime/pointcept/libs/pointops && python3 -m pip install --no-build-isolation .)
```

These commands have **not** been executed or validated as a fresh installation in this release. Resolve the unpinned package versions against the recorded Python/PyTorch/CUDA ABI before claiming exact environment replication. The server's current environment is not changed by the release scripts. The wrapper sets `PYTHONPATH` to the included runtime automatically.

LitePT additionally requires **FlashAttention 2.8.3** and **PointROPE 0.0.0**. The original build notes and wheel SHA256 values are in [litept-runtime-recorded.md](litept-runtime-recorded.md). The recorded FlashAttention wheel was built from the official 2.8.3 source with CUDA 12.6 against the older base image; its prebuilt upstream wheel was incompatible with that image's GLIBC. On a compatible fresh build environment:

```bash
python3 -m pip install --no-build-isolation flash-attn==2.8.3
(cd runtime/pointcept/libs/pointrope && python3 -m pip install --no-build-isolation .)
```

A new wheel build need not have the historical wheel hash. Exact historical wheel verification requires the original private environment artifacts. PTv3 uses `enable_flash=False`; LitePT requires its own recorded flash/rope dependencies. PTv1/PTv2 use the included PointOps extension. SpUNet, OA-CNNs, PTv3 and LitePT use spconv. PointNet++ uses its included PyTorch MSG implementation and the common Pointcept transform/loss code; the selected runtime shares imports, so use the common environment.

On **2026-11-09**, the released packages, data adapters, model registry and training/test engines were actually imported from the public `runtime/` tree in an existing benchmark image. The seven representative `seed42.py` files were parsed with the real `Config.fromfile`, and all seven models were constructed on CPU. All 66 relevant imported local modules resolved inside the public runtime; no old project source was mounted. The installed PointOps extension imported, and its nine Python source files matched the included extension source.

The base image lacks FlashAttention and PointROPE. Their already-installed files were copied read-only from the stopped six-class benchmark container into private temporary storage and mounted read-only for a separate LitePT check. FlashAttention 2.8.3 and PointROPE 0.0.0 then imported successfully, and LitePT CPU construction passed. No packages were installed, no historical container was restarted, and no image was rebuilt. The observed environment uses Python 3.10.20, torch 2.5.0 (compiled CUDA 12.4), spconv 2.3.8 and PointOps 1.0; further observed versions and the exact verification scope are in [runtime_check_20261109.json](runtime_check_20261109.json).

These checks establish package import and CPU-construction compatibility in the inspected existing environment. Fresh installation, CUDA kernel execution, model forward passes, training and formal ID/OOD inference were **not** run. The complete historical environment lock remains unavailable; the observed versions are not substituted for unknown versions in other historical runs.

No Git commit was recorded in the supplied source trees. [source_versions.json](source_versions.json) identifies each unmodified selected backbone by SHA256. Source trees are reduced by explicit file selection and restricted package registries; unused heads and optimizer imports are omitted. Selected forward passes, losses, transforms, optimizers and metric formulas are preserved.
