#!/usr/bin/env python3
"""
Run per-generator (attack-wise) evaluation for Default variant runs recorded in an ablation results folder.

This script reads <results_dir>/ablation_results.csv, finds rows with variant=='default',
and for each completed run it filters the provided metadata by generator and calls `evaluate.py`
to produce per-generator `metrics.json` files under the run's eval directory.

Usage:
  python scripts/run_attackwise_default.py --results_dir results_module_ablation --metadata metadata_kaggle.csv --features lfcc_features_kaggle
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_module_ablation')
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated seeds to restrict to')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    agg_csv = res_dir / 'ablation_results.csv'
    if not agg_csv.exists():
        print('ERROR: ablation_results.csv not found in', res_dir)
        return

    df = pd.read_csv(agg_csv)
    df = df[df['variant'] == 'default']
    if args.seeds:
        seeds = {int(s) for s in args.seeds.split(',')}
        df = df[df['seed'].isin(seeds)]

    meta_df = pd.read_csv(args.metadata)
    gens = sorted(meta_df['generator'].unique())

    PY = sys.executable
    repo_root = Path(__file__).resolve().parents[1]
    eval_script = repo_root / 'evaluate.py'

    for _, row in df.iterrows():
        run_dir = Path(row['run_dir'])
        variant = row['variant']
        seed = int(row['seed'])
        ckpt = run_dir / variant / f"seed_{seed}" / f"best_specrnet_{variant}_seed{seed}.pt"
        if not ckpt.exists():
            print(f"Skipping run (checkpoint not found): {run_dir} variant={variant} seed={seed}")
            continue

        for g in gens:
            out_eval = run_dir / variant / f"seed_{seed}" / 'eval' / f"attack_{g}"
            out_eval.mkdir(parents=True, exist_ok=True)
            submeta = meta_df[meta_df['generator'] == g]
            submeta_path = out_eval / 'metadata_attack.csv'
            if submeta_path.exists() and not args.overwrite:
                print(f"Skipping existing attack eval for {run_dir} seed={seed} attack={g}")
                continue
            submeta.to_csv(submeta_path, index=False)

            cmd = [PY, str(eval_script), '--checkpoint', str(ckpt), '--metadata', str(submeta_path), '--features', args.features, '--split', args.split, '--output_dir', str(out_eval), '--variant', variant, '--seed', str(seed)]
            print('Running:', ' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed for {run_dir} seed={seed} attack={g}: {e}")


if __name__ == '__main__':
    main()
