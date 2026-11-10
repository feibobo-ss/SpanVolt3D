#!/usr/bin/env python3
"""Portable launcher for the selected, existing seven-model implementations.

No-argument invocation prints help. `inspect` and `--dry-run` never import Torch,
read point arrays, create output files, launch training, or run inference.
"""
import argparse
import json
import os
from pathlib import Path
import pprint
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('pointnet2', 'spunet', 'oacnns', 'ptv1', 'ptv2', 'ptv3', 'litept')
DOMAINS = {'test_id': 'test', 'test_ood_110': 'ood_110', 'test_ood_1000': 'ood_1000'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('inspect', 'train', 'test'), default='inspect')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--seed', type=int, choices=(42, 3407, 9173), default=42)
    p.add_argument('--data-root', type=Path, required=True,
                   help='Already prepared six-class Nx7 NPY root, NOT native LAS')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--domain', choices=tuple(DOMAINS), default='test_id')
    p.add_argument('--checkpoint', type=Path,
                   help='For test: this model/seed validation-selected model_best.pth')
    p.add_argument('--dry-run', action='store_true', help='Print configuration and command only')
    return p


def load_config(model, seed):
    return {k: v for k, v in runpy.run_path(str(ROOT / 'configs' / model / f'seed{seed}.py')).items()
            if not k.startswith('__')}


def main():
    p = parser()
    if len(sys.argv) == 1:
        p.print_help()
        return
    args = p.parse_args()
    if args.action == 'test' and args.checkpoint is None:
        p.error('--checkpoint is required for test; use the validation-selected checkpoint')
    cfg = load_config(args.model, args.seed)
    data_root, output = args.data_root.resolve(), args.output.resolve()
    cfg['data_root'], cfg['save_path'] = str(data_root), str(output)
    cfg['weight'], cfg['resume'] = None, False
    for split in ('train', 'val', 'test'):
        cfg['data'][split]['data_root'] = str(data_root)
    if args.action == 'test':
        cfg['weight'] = str(args.checkpoint.resolve())
        cfg['test_only'], cfg['direct_test_output'] = True, True
        cfg['save_test_predictions'] = False
        cfg['data']['test']['split'] = DOMAINS[args.domain]
    resolved_config = output / 'resolved_config.py'
    command = []
    if args.model == 'pointnet2':
        script = 'train_las_final.py' if args.action == 'train' else 'test_las_final.py'
        command = [sys.executable, str(ROOT / 'runtime/pointnet2' / script),
                   '--config-file', str(resolved_config), '--in-channels', '3',
                   '--seed', str(args.seed), '--save-path', str(output)]
        if args.action == 'train':
            saved = json.loads((ROOT / 'configs/pointnet2' / f'seed{args.seed}_training.json').read_text())
            for name in ('model', 'epoch', 'eval_step', 'learning_rate', 'weight_decay', 'step_size', 'lr_decay', 'num_workers'):
                command += ['--' + name.replace('_', '-'), str(saved[name])]
        else:
            command += ['--split', DOMAINS[args.domain], '--checkpoint', str(args.checkpoint.resolve())] if args.checkpoint else []
    else:
        script = 'train.py' if args.action == 'train' else 'test.py'
        command = [sys.executable, str(ROOT / 'runtime/pointcept/tools' / script),
                   '--config-file', str(resolved_config), '--num-gpus', '1']
    report = dict(action=args.action, model=args.model, seed=args.seed,
                  input_representation='benchmark6: Nx7, XYZ columns 0:3, label column 6; RGB never used',
                  data_root=str(data_root), output=str(output),
                  data_passes=200, validation_every_passes=20,
                  model_config=({'implementation': 'pointnet2_sem_seg_msg_las_final', 'in_channels': 3, 'num_classes': 6}
                                if args.model == 'pointnet2' else cfg['model']),
                  command=None if args.action == 'inspect' else shlex.join(command))
    print(json.dumps(report, indent=2))
    if args.action == 'inspect' or args.dry_run:
        return
    expected_splits = ('train', 'val', 'test') if args.action == 'train' else (DOMAINS[args.domain],)
    for split in expected_splits:
        if not any((data_root / split).glob('*.npy')):
            p.error(f'No six-class NPY files found in {data_root / split}')
    if args.action == 'test' and not args.checkpoint.is_file():
        p.error('Checkpoint file does not exist')
    if output.exists() and any(output.iterdir()):
        p.error('Output directory must be empty: use a new directory for each run/domain')
    output.mkdir(parents=True, exist_ok=True)
    resolved_config.write_text('\n'.join(f'{k} = {pprint.pformat(v, width=100, sort_dicts=False)}' for k, v in cfg.items()) + '\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'runtime/pointcept'), str(ROOT / 'runtime/pointnet2'), env.get('PYTHONPATH', '')])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
