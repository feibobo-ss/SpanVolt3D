# LitePT runtime lock

- Base image: `pointcept_snapshot:20260519_204536`
- PyTorch: 2.5.0+cu124
- GPU target: NVIDIA RTX 4090
- FlashAttention: 2.8.3, built from the official PyPI source archive with
  CUDA 12.6 against the base image to preserve its older GLIBC compatibility
- PointROPE: repository `runtime/pointcept/libs/pointrope`, CUDA extension build

SHA-256:

- `flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`:
  `59f7480d9fe5a04305a1aac74cbe6090ca3c6410a536da2a4fc3c33ef44ab46a`
- `pointrope-0.0.0-cp310-cp310-linux_x86_64.whl`:
  `e09d7ecd9ebd67f626b2392a289fef20745b6f0898cdd1cfad41c94e5b9e87e5`

The upstream prebuilt FlashAttention wheel required GLIBC 2.32 and was not
used. It was retained only in the private source workspace as an environment
diagnostic artifact and is not distributed here.
