# Seven models, three recorded seeds

Each model directory contains the actual saved seed 42, 3407 and 9173 configuration for the 400-scene, six-class, XYZ-only run dated 2026-09-08. Only data and output paths were replaced with portable defaults. PointNet++ also includes the actual training argument JSON for each seed. Its Python configuration is the recorded shared data-transform configuration; network selection and optimization are supplied by that argument JSON and the PointNet++ trainer, so unused Pointcept model fields in this data configuration do not define the PointNet++ network. Configurations from the unrelated Ground7 run are excluded.

| Model | Selected implementation | Specific variant |
|---|---|---|
| PointNet++ | `pointnet2_sem_seg_msg_las_final` | MSG, external XYZ=3 channels; first grouping concatenates relative XYZ, giving 6 geometric channels |
| SpUNet | Pointcept `SpUNet-v1m1` | `spconv_unet_v1m1_base.py`; `spconv.pytorch` backend |
| OA-CNNs | Pointcept `OACNNs` | `oacnns_v1m1_base.py` |
| PTv1 | Pointcept `PointTransformer-Seg50` | `point_transformer_seg.py` |
| PTv2 | Pointcept `PT-v2m2` | Mode 2/base; `unpool_backend='interp'` |
| PTv3 | Pointcept `PT-v3m1` | `enable_flash=False` |
| LitePT | Pointcept `LitePT-v1` | **LitePT-S, small v1m1 lightweight decoder**; encoder widths 36/72/144/252/504, depths 2/2/2/6/2, head width 72, decoder depths all zero |

The original LitePT family config explicitly identifies `LitePT-S` and its v1m1 lightweight decoder; the saved run widths/depths match it. The exact implementation is included, with SHA256 identity in `environments/source_versions.json`. These sources are not replaced by newer upstream revisions.

For Pointcept, `epoch=200`, `eval_epoch=10` and `train.loop=20` mean 10 training loop epochs, each making 20 data passes, with validation every 20 passes. `default_config_parser` recomputes `train.loop=epoch//eval_epoch`. PointNet++ independently uses 200 epochs with dataset loop 1 and `eval_step=20`. PointNet++ uses FP32 and Adam; other models retain their own recorded AMP, optimizer and scheduler settings.

All inputs are six-class, already preprocessed Nx7 NPY arrays. Model input features are XYZ only. RGB storage slots are ignored. The 0.08 m model grid and 60,000-point training crop/test chunk are distinct from the 0.30 m data preprocessing voxel.

The examples below document the historical six-class benchmark. Complete reproduction requires all of its fixed data splits. [Current data access](../DATA_ACCESS.md) provides only approved native LAS project/scene subsets for non-commercial research by universities and research institutions, without republication of the supplied data; it does not include the complete corpus or frozen six-class inputs. A received native subset can be used with its original labels for the approved purpose, but is not directly the input to these benchmark commands. The full-corpus preparation and verification scripts do not accept a shortened manifest. Command inspection and `--dry-run` remain usable without point-cloud data.

Use the portable launcher from the repository root (the first two commands do not run a model):

```bash
python3 scripts/run_benchmark.py --help
python3 scripts/run_benchmark.py --action train --model ptv3 --seed 42 --data-root /path/to/benchmark6 --output runs/ptv3/seed42 --dry-run
python3 scripts/run_benchmark.py --action train --model ptv3 --seed 42 --data-root /path/to/benchmark6 --output runs/ptv3/seed42
python3 scripts/run_benchmark.py --action test --model ptv3 --seed 42 --data-root /path/to/benchmark6 --output runs/ptv3/seed42/eval_ood_110 --domain test_ood_110 --checkpoint runs/ptv3/seed42/model/model_best.pth
```

For the Pointcept commands above, replace `ptv3` with `spunet`, `oacnns`, `ptv1`, `ptv2`, or `litept` as needed. Test domains are `test_id`, `test_ood_110`, `test_ood_1000`, mapped to stored data folders `test`, `ood_110`, `ood_1000`. Keep the same validation-selected checkpoint for all domains. No checkpoint is included in this repository.

PointNet++ requires explicit tests for all three domains. Its historical result reader expects `test_test`, `test_ood_110`, and `test_ood_1000` output directories, each containing `summary.json`. Use these separate commands for seed 42; append `--dry-run` to inspect each command without inference:

```bash
python3 scripts/run_benchmark.py --action test --model pointnet2 --seed 42 --data-root /path/to/benchmark6 --output runs/pointnet2/seed42/test_test --domain test_id --checkpoint runs/pointnet2/seed42/checkpoints/model_best.pth
python3 scripts/run_benchmark.py --action test --model pointnet2 --seed 42 --data-root /path/to/benchmark6 --output runs/pointnet2/seed42/test_ood_110 --domain test_ood_110 --checkpoint runs/pointnet2/seed42/checkpoints/model_best.pth
python3 scripts/run_benchmark.py --action test --model pointnet2 --seed 42 --data-root /path/to/benchmark6 --output runs/pointnet2/seed42/test_ood_1000 --domain test_ood_1000 --checkpoint runs/pointnet2/seed42/checkpoints/model_best.pth
```

For seed 3407 or 9173, change `--seed` and the corresponding seed directory in both `--output` and `--checkpoint` together. The PointNet++ run seed is supplied by `--seed`; unused seed/model fields in its recorded shared data configuration are not a reason to edit that configuration. The training launcher also accepts `--model pointnet2`; its best checkpoint is saved under `checkpoints/model_best.pth`.

Pointcept training automatically invokes its ID `PreciseEvaluator`. The historical `evaluation/aggregate.py --run-root` reader obtains ID scores from that post-training evaluation in `<model>/seed<seed>/train.log`. A separately launched Pointcept ID test writes `eval.txt` in its own `--output` directory, which is a different collection path. `--run-root` accepts the documented historical tree, including Pointcept OOD files `eval_ood_110/eval.txt` and `eval_ood_1000/eval.txt`; it does not discover arbitrary output directories or collect a standalone ID `eval.txt` automatically. See [evaluation details](../docs/evaluation_details.md) for the exact layout.

The wrapper defaults to `inspect` and prints help with no arguments. Actual `train`/`test` actions require a new empty output directory, materialize only the selected config with supplied paths, and invoke the included original engines. `--dry-run` reads only source configs, never point arrays or weights. The original checkpoint selection differs only on ties: Pointcept retains the first strict maximum (`>`); PointNet++ updates on equal maximum (`>=`). Both select using validation mIoU, with no OOD selection.
