"""Aggregate ablation results from results/<variant>/seed_<seed>/summary.json
Produces a CSV summary with mean/std for selected metrics across seeds.
"""
import json
from pathlib import Path
import argparse
import statistics
import csv

def collect_summaries(results_dir: Path):
    rows = {}
    for variant_dir in results_dir.iterdir():
        if not variant_dir.is_dir() or variant_dir.name.startswith('eval_test'):
            continue
        runs = []
        for seed_dir in variant_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            run_data = {}
            summary_file = seed_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    run_data.update(data)
            
            # 注意：由于我们的 evaluate.py 目前没有直接写 metrics.json，
            # 这里主要汇总 summary.json 中的 params, precise_ms 和 val_best_f1
            if run_data:
                runs.append(run_data)
        if runs:
            rows[variant_dir.name] = runs
    return rows

def summarize(rows):
    out = []
    for variant, items in rows.items():
        params = [r.get('params') for r in items if r.get('params') is not None]
        precise_ms = [r.get('precise_ms') for r in items if r.get('precise_ms') is not None]
        best_f1 = [r.get('best_f1') for r in items if r.get('best_f1') is not None]

        row = {
            'variant': variant,
            'runs': len(items),
            'val_f1_mean': statistics.mean(best_f1) if best_f1 else None,
            'params_mean': int(statistics.mean(params)) if params else None,
            'precise_ms_mean': statistics.mean(precise_ms) if precise_ms else None,
        }
        out.append(row)
    return out

def main():
    parser = argparse.ArgumentParser()
    # 改为新的 kaggle 结果路径
    parser.add_argument('--results_dir', type=str, default='results_kaggle_balanced')
    parser.add_argument('--out_csv', type=str, default='results_kaggle_balanced/ablation_summary.csv')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    rows = collect_summaries(results_dir)
    summary = summarize(rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not summary:
        print("No summary data found.")
        return

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
        
    print(f"✅ Aggregation complete. Saved to {out_path}")

if __name__ == '__main__':
    main()