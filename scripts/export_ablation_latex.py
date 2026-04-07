#!/usr/bin/env python3
"""
Export aggregated ablation CSVs to LaTeX tables and save backups under reports/latex/.
Writes per-dataset: ablation_summary_table.tex and an attack-wise table if available.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

TASK_ROOT = Path(os.environ.get('TASK_RESULTS_ROOT', 'results_task_ablation'))
REPO = Path('.').resolve()
REPORTS_DIR = REPO / 'reports' / 'latex'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ('Module Ablation', REPO / 'results_module_ablation' / 'analysis_summary_by_variant.csv', REPO / 'results_module_ablation'),
    ('Short Utterance', TASK_ROOT / 'short_utterance' / 'analysis_summary_by_variant.csv', TASK_ROOT / 'short_utterance'),
    ('Data Scarcity', TASK_ROOT / 'data_scarcity' / 'analysis_summary_by_variant.csv', TASK_ROOT / 'data_scarcity'),
]

METRIC_ORDER = ['eval_f1', 'eval_auc', 'eval_eer']
METRIC_NAMES = {'eval_f1':'F1', 'eval_auc':'AUC', 'eval_eer':'EER'}


def fmt_num(mean, std):
    try:
        mean = float(mean)
    except Exception:
        return ''
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"${mean:.4f}$"
    try:
        std = float(std)
        return f"${mean:.4f}\\pm{std:.4f}$"
    except Exception:
        return f"${mean:.4f}$"


def escape_latex(s):
    return str(s).replace('_','\\_')


def export_table(df: pd.DataFrame, metrics, out_path: Path, caption: str):
    # ensure variant and n
    if 'variant' not in df.columns:
        df = df.copy()
        df['variant'] = 'unknown'
    if 'n' not in df.columns:
        df = df.copy()
        df['n'] = df.index.to_series().apply(lambda x: 0)

    cols = ['Variant', 'n'] + [METRIC_NAMES.get(m, m) for m in metrics]
    col_spec = 'l ' + 'c ' * (1 + len(metrics))

    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append(f"\\begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append('\\hline')
    lines.append(' & '.join(cols) + ' \\\\')
    lines.append('\\hline')

    for _, r in df.iterrows():
        variant = escape_latex(r.get('variant',''))
        n = int(r.get('n',0)) if not pd.isna(r.get('n', None)) else 0
        cells = [variant, str(n)]
        for m in metrics:
            mean_k = f"{m}_mean"
            std_k = f"{m}_std"
            mean = r.get(mean_k, np.nan)
            std = r.get(std_k, np.nan) if std_k in r.index else None
            cells.append(fmt_num(mean, std))
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append(f"\\caption{{{caption}}}")
    label = caption.lower().replace(' ', '_')
    lines.append(f"\\label{{tab:{label}}}")
    lines.append('\\end{table}')

    out_path.write_text('\n'.join(lines))


def main():
    for title, csv_path, out_dir in DATASETS:
        print(f"Processing: {title} -> {csv_path}")
        if not csv_path.exists():
            print(f"  [SKIP] CSV not found: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [ERROR] Failed reading {csv_path}: {e}")
            continue

        # determine metrics present
        metrics = [m for m in METRIC_ORDER if f"{m}_mean" in df.columns]
        if not metrics:
            metrics = sorted({c[:-5] for c in df.columns if c.endswith('_mean')})
        if not metrics:
            print(f"  [WARN] No *_mean metric columns found in {csv_path}")
            continue

        target_dir = Path(out_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        target_tex = target_dir / 'ablation_summary_table.tex'
        caption = f"{title} summary (mean $\\pm$ std)"
        export_table(df, metrics, target_tex, caption)
        # backup to repo reports
        backup_tex = REPORTS_DIR / (title.lower().replace(' ', '_') + '_ablation_table.tex')
        backup_tex.write_text(target_tex.read_text())
        print(f"  Wrote: {target_tex} and backup: {backup_tex}")

    # attack-wise
    attack_csv = REPO / 'results_module_ablation' / 'ablation_attackwise_default.csv'
    if attack_csv.exists():
        txt = attack_csv.read_text()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        data_rows = []
        for l in lines:
            parts = [p.strip() for p in l.split(',')]
            if len(parts) < 4:
                continue
            try:
                mean = float(parts[1])
                std = float(parts[2])
                n = int(float(parts[3]))
                data_rows.append((parts[0], mean, std, n))
            except Exception:
                continue
        if data_rows:
            lines = []
            lines.append('\\begin{table}[ht]')
            lines.append('\\centering')
            lines.append('\\small')
            lines.append('\\begin{tabular}{l c c}')
            lines.append('\\hline')
            lines.append('Attack & n & EER \\\\')
            lines.append('\\hline')
            for attack, mean, std, n in data_rows:
                lines.append(f"{escape_latex(attack)} & {n} & ${mean:.4f}\\pm{std:.4f}$ \\\\\n")
            lines.append('\\hline')
            lines.append('\\end{tabular}')
            lines.append('\\caption{Attack-wise EER for Default.}')
            lines.append('\\label{tab:attackwise_default}')
            lines.append('\\end{table}')
            out_attack = REPO / 'results_module_ablation' / 'attackwise_default_table.tex'
            out_attack.write_text('\n'.join(lines))
            backup_attack = REPORTS_DIR / 'attackwise_default_table.tex'
            backup_attack.write_text(out_attack.read_text())
            print(f"  Wrote attack-wise table: {out_attack} and backup: {backup_attack}")
        else:
            print('  [INFO] No numeric attack rows parsed from attackwise CSV.')
    else:
        print('  [INFO] No attackwise CSV present.')

    print('\nExport complete.')

if __name__ == '__main__':
    main()
