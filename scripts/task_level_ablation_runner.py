#!/usr/bin/env python3
"""
Orchestrate Task-level ablations:
- Short Utterance Ablation: extract 1s LFCCs (padded to network input) and run experiments.
- Data Scarcity Ablation: build 10%%-train metadata and run experiments.

This script only composes commands and can perform a dry-run.

Usage examples:
  python scripts/task_level_ablation_runner.py --metadata metadata_kaggle.csv --features lfcc_features_kaggle --out_root results_task_ablation --dry_run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable

def run(cmd, dry_run=False, env=None):
    print('CMD:', ' '.join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--out_root', type=str, default='results_task_ablation')
    parser.add_argument('--seeds', type=str, default='42,7,123')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--train_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Short utterance: extract 1s features to a new folder
    short_feat_dir = out_root / 'lfcc_features_1s'
    cmd_extract = [PY, str(Path(__file__).resolve().parents[1] / 'extract_lfcc.py'), '--metadata_csv', args.metadata, '--output_dir', str(short_feat_dir), '--duration', '1.0', '--n_lfcc', '80', '--n_frames', '404']
    # Data scarcity: build 10% train metadata
    scarcity_meta = out_root / (Path(args.metadata).stem + f'_train{int(args.train_frac*100)}pct.csv')
    cmd_build_meta = [PY, str(Path(__file__).resolve().parents[1] / 'scripts' / 'build_metadata_scarcity.py'), '--metadata', args.metadata, '--out_csv', str(scarcity_meta), '--train_frac', str(args.train_frac), '--seed', str(args.seed)]

    # hparam ablation base cmd (we run variants default,no-att,gap as module ablation)
    hparam = str(Path(__file__).resolve().parents[1] / 'scripts' / 'hparam_ablation.py')

    # Short utterance experiment
    short_out = out_root / 'short_utterance'
    short_cmd = [PY, hparam, '--metadata', args.metadata, '--features', str(short_feat_dir), '--out_dir', str(short_out), '--variants', 'default,no-att,gap', '--lrs', '1e-4', '--batch_sizes', '128', '--seeds', args.seeds, '--epochs', str(args.epochs), '--run_eval']

    # Data scarcity experiment
    scarcity_out = out_root / 'data_scarcity'
    scarcity_cmd = [PY, hparam, '--metadata', str(scarcity_meta), '--features', args.features, '--out_dir', str(scarcity_out), '--variants', 'default,no-att,gap', '--lrs', '1e-4', '--batch_sizes', '128', '--seeds', args.seeds, '--epochs', str(args.epochs), '--run_eval']

    env = os.environ.copy()

    # 1) extract short features
    rc = run(cmd_extract, dry_run=args.dry_run, env=env)
    if rc != 0:
        print('Failed extracting short features')
        return rc

    # 2) run short utterance ablation (dry-run possible)
    rc = run(short_cmd, dry_run=args.dry_run, env=env)
    if rc != 0:
        print('Short utterance ablation failed')
        return rc

    # 3) build scarcity metadata
    rc = run(cmd_build_meta, dry_run=args.dry_run, env=env)
    if rc != 0:
        print('Failed building scarcity metadata')
        return rc

    # 4) run data scarcity ablation
    rc = run(scarcity_cmd, dry_run=args.dry_run, env=env)
    if rc != 0:
        print('Data scarcity ablation failed')
        return rc

    print('Task-level ablations scheduled/completed under', str(out_root))
    return 0

if __name__ == '__main__':
    sys.exit(main())
