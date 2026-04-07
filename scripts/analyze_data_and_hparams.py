#!/usr/bin/env python3
"""
Analyze metadata CSVs (training data distribution) and ablation/hyperparameter CSVs.
Saves CSV summaries under `reports/csv/` and prints concise summaries.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path('.').resolve()
TASK_ROOT = Path(os.environ.get('TASK_RESULTS_ROOT', str(REPO / 'results_task_ablation')))
REPORTS = REPO / 'reports' / 'csv'
REPORTS.mkdir(parents=True, exist_ok=True)

METADATA_GLOB = ['metadata*.csv']
ABlation_GLOB = ['**/ablation_results*.csv']

METRICS = ['eval_f1','eval_auc','eval_eer']


def safe_read(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] failed reading {p}: {e}")
        return None


def summarize_metadata_files():
    files = list(REPO.glob('metadata*.csv')) + list(REPO.glob('**/metadata*.csv'))
    # dedupe preserving order
    seen = set(); uniq = []
    for f in files:
        if str(f.resolve()) not in seen:
            seen.add(str(f.resolve())); uniq.append(f)
    if not uniq:
        print('[INFO] No metadata CSVs found in repo root or subdirs.')
    rows = []
    for p in uniq:
        df = safe_read(p)
        if df is None:
            continue
        total = len(df)
        splits = df['split'].value_counts(dropna=False).to_dict() if 'split' in df.columns else {}
        labels = df['label'].value_counts(dropna=False).to_dict() if 'label' in df.columns else {}
        generators = df['generator'].value_counts(dropna=False).to_dict() if 'generator' in df.columns else {}
        utt_unique = df['utt_id'].nunique() if 'utt_id' in df.columns else None
        rows.append({'file': str(p), 'total': total, 'unique_utt': utt_unique, 'n_generators': len(generators)})
        # save per-file breakdowns
        if splits:
            pd.Series(splits).rename('count').to_frame().to_csv(REPORTS / (p.stem + '_split_counts.csv'))
        if labels:
            pd.Series(labels).rename('count').to_frame().to_csv(REPORTS / (p.stem + '_label_counts.csv'))
        if generators:
            pd.Series(generators).rename('count').to_frame().to_csv(REPORTS / (p.stem + '_generator_counts.csv'))
    if rows:
        pd.DataFrame(rows).to_csv(REPORTS / 'metadata_summary.csv', index=False)
        print('\nMetadata summary written to reports/csv/metadata_summary.csv')
        for r in rows:
            print(f" - {Path(r['file']).name}: total={r['total']}, unique_utt={r['unique_utt']}, generators={r['n_generators']}")
    else:
        print('[INFO] No metadata summaries produced.')


def normalize_cols(df):
    cols = {c:c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    # common aliases
    if 'bs' in df.columns and 'batch_size' not in df.columns:
        df = df.rename(columns={'bs':'batch_size'})
    if 'lr' in df.columns and 'lr' not in df.columns:
        pass
    return df


def agg_stats(df, groupby, metrics=METRICS):
    present = [m for m in metrics if m in df.columns]
    if not present:
        return None
    agg = df.groupby(groupby, dropna=False)[present].agg(['mean','std','count'])
    # flatten
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    return agg


def analyze_ablation_files():
    candidates = list(REPO.glob('results_module_ablation/ablation_results.csv'))
    # include any ablation_results files anywhere
    candidates += list(REPO.glob('**/ablation_results.csv'))
    # include task-root ones
    candidates += [TASK_ROOT / 'short_utterance' / 'ablation_results.csv', TASK_ROOT / 'data_scarcity' / 'ablation_results.csv']
    # dedupe
    uniq = []
    seen = set()
    for p in candidates:
        if p.exists():
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp); uniq.append(p)
    if not uniq:
        print('[INFO] No ablation_results CSVs found.')
        return
    for p in uniq:
        df = safe_read(p)
        if df is None:
            continue
        df = normalize_cols(df)
        # cast numeric
        for m in METRICS:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors='coerce')
        # metrics by variant
        if 'variant' in df.columns:
            outv = agg_stats(df, ['variant'])
            if outv is not None:
                outv.to_csv(REPORTS / (p.stem + '_' + 'by_variant.csv'), index=False)
        # metrics by lr
        if 'lr' in df.columns:
            df['lr'] = pd.to_numeric(df['lr'], errors='coerce')
            outr = agg_stats(df, ['lr'])
            if outr is not None:
                outr.to_csv(REPORTS / (p.stem + '_' + 'by_lr.csv'), index=False)
        # metrics by batch_size
        if 'batch_size' in df.columns:
            df['batch_size'] = pd.to_numeric(df['batch_size'], errors='coerce')
            outb = agg_stats(df, ['batch_size'])
            if outb is not None:
                outb.to_csv(REPORTS / (p.stem + '_' + 'by_batch.csv'), index=False)
        # metrics by lfcc
        if 'lfcc' in df.columns:
            df['lfcc'] = pd.to_numeric(df['lfcc'], errors='coerce')
            outl = agg_stats(df, ['lfcc'])
            if outl is not None:
                outl.to_csv(REPORTS / (p.stem + '_' + 'by_lfcc.csv'), index=False)
        # combined variant+lr+batch
        group_cols = [c for c in ['variant','lr','batch_size'] if c in df.columns]
        if group_cols:
            outc = agg_stats(df, group_cols)
            if outc is not None:
                outc.to_csv(REPORTS / (p.stem + '_' + 'by_variant_lr_batch.csv'), index=False)
        print(f"Analyzed {p} -> saved summaries to reports/csv/ (prefix {p.stem}_*)")


def main():
    print('Starting analysis: metadata + ablation/hparams')
    summarize_metadata_files()
    analyze_ablation_files()
    print('\nAll analysis outputs are under reports/csv/')

if __name__ == '__main__':
    main()
