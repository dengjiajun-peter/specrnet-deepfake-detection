#!/usr/bin/env python3
"""
Compute aggregate summaries (mean ± std) for ablation CSVs.
Saves per-dataset summary CSVs and prints concise tables.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

METRICS = ['eval_f1', 'eval_auc', 'eval_eer']


def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")
        return None


def summarize_by_variant(df: pd.DataFrame, metrics=METRICS):
    cols_present = [c for c in metrics if c in df.columns]
    if 'variant' not in df.columns:
        # try to infer variant column name
        for c in ['variant', 'variant_name', 'model']:
            if c in df.columns:
                df = df.rename(columns={c: 'variant'})
                break
    if 'variant' not in df.columns:
        df['variant'] = 'unknown'
    for c in cols_present:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    groups = []
    gobj = df.groupby('variant', dropna=False)
    for name, g in gobj:
        row = {'variant': name, 'n': len(g)}
        for c in cols_present:
            vals = g[c].dropna().astype(float)
            row[f'{c}_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f'{c}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        groups.append(row)
    return pd.DataFrame(groups)


def summarize_attackwise(df: pd.DataFrame):
    # try to find an attack column and an EER-like column
    attack_col = None
    eer_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ('attack','generator','attack_name'):
            attack_col = c
        if 'eer' in lc:
            eer_col = c
    if attack_col is None:
        print('[WARN] No attack/generator column found in attackwise CSV; skipping attackwise summary')
        return None
    if eer_col is None:
        print('[WARN] No EER-like column found in attackwise CSV; trying any numeric column')
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            eer_col = numeric_cols[-1]
        else:
            return None
    df[eer_col] = pd.to_numeric(df[eer_col], errors='coerce')
    g = df.groupby(attack_col, dropna=False)[eer_col]
    out = g.agg(['mean','std','count']).reset_index()
    out = out.rename(columns={'mean':'eer_mean','std':'eer_std','count':'n'})
    return out


def main():
    repo = Path('.').resolve()
    # Allow overriding task-level results root (may be outside repo) via env var TASK_RESULTS_ROOT
    task_root = os.environ.get('TASK_RESULTS_ROOT', str(repo / 'results_task_ablation'))
    datasets = [
        ('Module Ablation', repo / 'results_module_ablation' / 'ablation_results.csv', repo / 'results_module_ablation'),
        ('Short Utterance', Path(task_root) / 'short_utterance' / 'ablation_results.csv', Path(task_root) / 'short_utterance'),
        ('Data Scarcity', Path(task_root) / 'data_scarcity' / 'ablation_results.csv', Path(task_root) / 'data_scarcity'),
    ]

    for title, csv_path, out_dir in datasets:
        print('\n' + '='*40)
        print(f"Dataset: {title}")
        print('CSV:', csv_path)
        df = safe_read_csv(csv_path)
        if df is None:
            print('[WARN] CSV missing, skipping.')
            continue
        summary = summarize_by_variant(df)
        if summary is None or summary.empty:
            print('[WARN] No summary produced (empty).')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'analysis_summary_by_variant.csv'
        summary.to_csv(out_file, index=False)
        print(f"Wrote summary to: {out_file}")
        # print a concise table
        for _, r in summary.sort_values('variant').iterrows():
            parts = [f"variant={r['variant']}", f"n={int(r['n'])}"]
            for m in METRICS:
                mean_k = f'{m}_mean'
                std_k = f'{m}_std'
                if mean_k in r and not pd.isna(r[mean_k]):
                    mean = r[mean_k]
                    std = r[std_k] if std_k in r else np.nan
                    if pd.isna(std):
                        parts.append(f"{m}={mean:.4f}")
                    else:
                        parts.append(f"{m}={mean:.4f}±{std:.4f}")
            print('  - ' + ', '.join(parts))

    # attack-wise
    attackwise_path = repo / 'results_module_ablation' / 'ablation_attackwise_default.csv'
    if attackwise_path.exists():
        print('\n' + '='*40)
        print('Attack-wise (Default) summary:')
        try:
            adf = pd.read_csv(attackwise_path)
            atab = summarize_attackwise(adf)
            if atab is not None:
                out_attack = repo / 'results_module_ablation' / 'analysis_attackwise_default.csv'
                atab.to_csv(out_attack, index=False)
                print(f'Wrote attack-wise summary to: {out_attack}')
                for _, r in atab.sort_values(by='eer_mean').iterrows():
                    print(f"  - {r.tolist()[0]}: eer_mean={r['eer_mean']:.4f} ± { (r['eer_std'] if not pd.isna(r['eer_std']) else float('nan')):.4f } (n={int(r['n'])})")
            else:
                print('[WARN] attack-wise aggregation returned nothing')
        except Exception as e:
            print(f'[WARN] Failed to process attackwise CSV: {e}')
    else:
        print('\n[INFO] No attackwise CSV found at', attackwise_path)

    print('\nAll analyses complete.')

if __name__ == '__main__':
    main()
