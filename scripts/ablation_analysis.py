#!/usr/bin/env python3
"""
Compute mean ± std statistics for ablation results and for attack-wise EER for Default variant.

Writes:
- <results_dir>/ablation_summary_overall.csv  (per-variant mean/std for eval_f1/auc/eer)
- <results_dir>/ablation_attackwise_default.csv  (per-generator mean/std of EER for Default)

Usage:
  python scripts/ablation_analysis.py --results_dir results_module_ablation
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_module_ablation')
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    # support both the canonical file and the previous misspelled export
    candidates = [res_dir / 'ablation_results.csv', res_dir / 'ablation_results-modeule.csv']
    dfs = []
    for p in candidates:
        if p.exists():
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                print('Warning: failed to read', p)
    if not dfs:
        print('ERROR: no ablation results CSV found in', res_dir)
        return
    df = pd.concat(dfs, ignore_index=True)
    # coerce numeric
    for c in ['eval_f1', 'eval_auc', 'eval_eer']:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    metrics = df.groupby('variant').agg({'eval_f1':['mean','std'], 'eval_auc':['mean','std'], 'eval_eer':['mean','std']})
    out_overall = res_dir / 'ablation_summary_overall.csv'
    metrics.to_csv(out_overall)
    print('Wrote overall summary to', out_overall)

    # Attack-wise EER only for default
    rows = []
    default_df = df[df['variant']=='default']
    # deduplicate multiple records for the same run (some exports may contain duplicates)
    if not default_df.empty:
        default_df = default_df.drop_duplicates(subset=['run_dir','variant','seed'])
    # for each default run, look for per-attack metrics under run_dir/default/seed_X/eval/attack_*/metrics.json
    for _, r in default_df.iterrows():
        run_dir = Path(r['run_dir'])
        variant = r['variant']
        seed = int(r['seed'])
        eval_root = run_dir / variant / f"seed_{seed}" / 'eval'
        if not eval_root.exists():
            continue
        for attack_dir in eval_root.glob('attack_*'):
            metrics_file = attack_dir / 'metrics.json'
            if not metrics_file.exists():
                continue
            try:
                m = json.loads(metrics_file.read_text())
                eer = m.get('eer')
                rows.append({'attack': attack_dir.name.replace('attack_',''), 'seed': seed, 'eer': safe_float(eer)})
            except Exception:
                continue

    if rows:
        at_df = pd.DataFrame(rows)
        stats = at_df.groupby('attack').agg({'eer':['mean','std','count']})
        out_attack = res_dir / 'ablation_attackwise_default.csv'
        stats.to_csv(out_attack)
        print('Wrote attack-wise default EER summary to', out_attack)
    else:
        print('No per-attack metrics found for default runs; run run_attackwise_default.py first.')


if __name__ == '__main__':
    main()
