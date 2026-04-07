#!/usr/bin/env python3
"""
Monitor training logs for multiple runs and print/append new epoch metrics.
Usage:
  python scripts/monitor_training.py --runs results_kaggle_balanced/no-gru/seed_7 results_kaggle_balanced/no-gru/seed_123 --interval 5
"""
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

OUT_DIR = Path('reports') / 'csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_CSV = OUT_DIR / 'monitoring_epochs.csv'


def tail_new_rows(log_path: Path, last_count: int):
    if not log_path.exists():
        return [], last_count
    try:
        with log_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return [], last_count
    if len(lines) <= 1:
        return [], 0
    data = lines[1:]
    if last_count >= len(data):
        return [], len(data)
    new = data[last_count:]
    rows = []
    for line in new:
        line = line.strip()
        if not line:
            continue
        try:
            rec = next(csv.reader([line]))
        except Exception:
            continue
        if len(rec) < 2:
            continue
        # try to parse known columns
        try:
            epoch = int(rec[0])
        except Exception:
            epoch = None
        def safe_float(i, default=float('nan')):
            try:
                return float(rec[i])
            except Exception:
                return default
        if epoch is None:
            continue
        train_loss = safe_float(1)
        val_f1 = safe_float(2)
        val_auc = safe_float(3)
        val_eer = safe_float(4)
        val_acc = safe_float(5)
        inf_ms = safe_float(6)
        grad_norm = safe_float(7)
        rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_eer': val_eer,
            'val_acc': val_acc,
            'inf_ms': inf_ms,
            'grad_norm': grad_norm,
        })
    return rows, len(data)


def monitor(runs, interval=5.0):
    last_counts = {r: 0 for r in runs}
    try:
        while True:
            any_found = False
            for run in runs:
                log_path = Path(run) / 'train_log.csv'
                new_rows, new_count = tail_new_rows(log_path, last_counts.get(run, 0))
                if new_rows:
                    any_found = True
                for r in new_rows:
                    ts = datetime.utcnow().isoformat() + 'Z'
                    print(f"[{run}] {ts} Epoch {r['epoch']} | train_loss={r['train_loss']:.4f} val_f1={r['val_f1']:.4f} val_auc={r['val_auc']:.4f} val_eer={r['val_eer']:.4f} val_acc={r['val_acc']:.4f}")
                    # append to master CSV
                    write_header = not MASTER_CSV.exists()
                    with MASTER_CSV.open('a', newline='', encoding='utf-8') as mf:
                        writer = csv.writer(mf)
                        if write_header:
                            writer.writerow(['run','timestamp','epoch','train_loss','val_f1','val_auc','val_eer','val_acc','inf_ms','grad_norm'])
                        writer.writerow([run, ts, r['epoch'], r['train_loss'], r['val_f1'], r['val_auc'], r['val_eer'], r['val_acc'], r['inf_ms'], r['grad_norm']])
                last_counts[run] = new_count
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Monitor stopped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()
    monitor(args.runs, args.interval)
