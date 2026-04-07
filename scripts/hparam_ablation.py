#!/usr/bin/env python3
"""
Hyperparameter ablation driver for SpecRNet reproduction repo.

Behavior:
- Builds a grid over variants × lrs × batch_sizes × lfcc_dims × seeds.
- For each job it creates a unique run directory and calls `train_specrnet.py`.
- Optionally runs `evaluate.py` on the best checkpoint produced by training.
- Aggregates results into a single CSV: <out_dir_root>/ablation_results.csv

Usage example (dry-run):
python scripts/hparam_ablation.py --metadata metadata_multi.csv --features lfcc_features --out_dir results_ablation --dry_run
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PY = sys.executable

_lock = threading.Lock()

def parse_list(s, cast=str):
    return [cast(x) for x in s.split(',')] if isinstance(s, str) and s != '' else []

def sanitize_name(x):
    return str(x).replace('/', '_').replace('\\', '_').replace(' ', '_').replace('.', 'p').replace('-', 'm').replace(':','_')

def find_train_summary(run_base, variant, seed):
    p = Path(run_base) / variant / f"seed_{seed}" / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text()), str(p)
        except Exception:
            return None, str(p)
    p2 = Path(run_base) / variant / f"seed_{seed}" / "history.json"
    if p2.exists():
        try:
            h = json.loads(p2.read_text())
            best_f1 = max(h.get('val_f1', [])) if 'val_f1' in h else None
            return ({'best_f1': best_f1}, str(p2))
        except Exception:
            return None, str(p2)
    p3 = Path(run_base) / variant / f"seed_{seed}" / "train_log.csv"
    if p3.exists():
        try:
            with open(p3, 'r', newline='') as cf:
                reader = list(csv.DictReader(cf))
                if not reader:
                    return None, str(p3)
                best = max(reader, key=lambda r: float(r.get('val_f1', 0.0)))
                return ({
                    'best_f1': float(best.get('val_f1', 0.0)),
                    'val_auc': float(best.get('val_auc', 0.0)),
                    'val_eer': float(best.get('val_eer', 0.0)),
                    'val_acc': float(best.get('val_acc', 0.0))
                }, str(p3))
        except Exception:
            return None, str(p3)
    return None, None

def find_eval_metrics(run_base, variant, seed):
    p = Path(run_base) / variant / f"seed_{seed}" / "eval" / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text()), str(p)
        except Exception:
            return None, str(p)
    return None, None

def append_csv(agg_path, row, header):
    with _lock:
        existed = Path(agg_path).exists()
        with open(agg_path, 'a', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=header)
            if not existed:
                writer.writeheader()
            writer.writerow(row)

def build_command(python_exec, train_script, args_map, extra_flags):
    cmd = [python_exec, train_script]
    for k, v in args_map.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(k)
        else:
            cmd.extend([k, str(v)])
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features', type=str, default='lfcc_features',
                        help='Default features directory. If using LFCC-dim sweep, use --features_template instead.')
    parser.add_argument('--features_template', type=str, default=None,
                        help='Template with {lfcc} placeholder, e.g. lfcc_features_n{lfcc}')
    parser.add_argument('--out_dir', type=str, default='results_ablation')
    parser.add_argument('--train_script', type=str, default='train_specrnet.py')
    parser.add_argument('--evaluate_script', type=str, default='evaluate.py')
    parser.add_argument('--variants', type=str, default='default,no-att,gap',
                        help='Comma-separated variant names (must match train_specrnet choices)')
    parser.add_argument('--lrs', type=str, default='1e-5,1e-4,5e-4')
    parser.add_argument('--batch_sizes', type=str, default='8,32,64')
    parser.add_argument('--lfcc_dims', type=str, default='80',
                        help='Comma-separated LFCC dims. If you use features_template, the template will be formatted with each dim.')
    parser.add_argument('--seeds', type=str, default='42,7,123')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none','cosine'])
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of concurrent training jobs (use 1 for single GPU).')
    parser.add_argument('--cuda_devices', type=str, default=None,
                        help='Comma-separated GPU ids to cycle (e.g. "0,1"). If omitted, inherits environment.')
    parser.add_argument('--run_eval', action='store_true', help='Run evaluate.py after training for each job.')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--dry_run', action='store_true', help='Print commands only, do not execute.')
    parser.add_argument('--overwrite', action='store_true', help='Re-run jobs even if results exist.')
    parser.add_argument('--extra_train_args', type=str, default='', help='Extra args appended to train command (space-separated tokens).')
    args = parser.parse_args()

    variants = [v for v in args.variants.split(',') if v]
    lrs = parse_list(args.lrs, float)
    batch_sizes = parse_list(args.batch_sizes, int)
    lfcc_dims = parse_list(args.lfcc_dims, int)
    seeds = parse_list(args.seeds, int)
    cuda_devices = parse_list(args.cuda_devices, str) if args.cuda_devices else []

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    agg_csv = out_base / 'ablation_results.csv'
    header = [
        'run_id','variant','seed','lr','batch_size','lfcc','train_features','run_dir',
        'train_summary_file','best_f1','precise_ms','params','train_returncode',
        'eval_metrics_file','eval_f1','eval_auc','eval_eer','eval_accuracy','eval_precision','eval_recall',
        'started_at','finished_at'
    ]

    jobs = []
    for variant in variants:
        for lr in lrs:
            for bs in batch_sizes:
                for lfcc in lfcc_dims:
                    if args.features_template:
                        features_path = args.features_template.format(lfcc=lfcc)
                    else:
                        features_path = args.features
                        if len(lfcc_dims) > 1:
                            candidate = f"{args.features}_{lfcc}"
                            if Path(candidate).exists():
                                features_path = candidate
                    if not Path(features_path).exists():
                        print(f"Skipping job: features path missing: {features_path}")
                        continue
                    for seed in seeds:
                        run_id = sanitize_name(f"{variant}_lr{lr}_bs{bs}_lfcc{lfcc}_s{seed}")
                        job_out_root = out_base / run_id
                        jobs.append({
                            'variant': variant, 'lr': lr, 'bs': bs, 'lfcc': lfcc, 'seed': seed,
                            'features': features_path, 'run_id': run_id, 'run_out': str(job_out_root)
                        })

    if not jobs:
        print("No jobs scheduled (empty job list). Check features paths / args.")
        return

    print(f"Scheduled {len(jobs)} jobs. max_workers={args.max_workers}. dry_run={args.dry_run}")

    def worker(job_idx, job):
        started_at = datetime.utcnow().isoformat()
        variant = job['variant']; lr = job['lr']; bs = job['bs']; lfcc = job['lfcc']; seed = job['seed']
        features_path = job['features']; run_id = job['run_id']; run_out = job['run_out']
        os.makedirs(run_out, exist_ok=True)
        cmd_args = {
            '--variant': variant,
            '--seed': seed,
            '--metadata': args.metadata,
            '--features': features_path,
            '--out_dir': run_out,
            '--batch_size': bs,
            '--epochs': args.epochs,
            '--lr': lr,
            '--weight_decay': args.weight_decay,
            '--scheduler': args.scheduler
        }
        extra_flags = ['--weighted_loss'] if args.weighted_loss else []
        if args.extra_train_args:
            extra_flags += args.extra_train_args.split()

        cmd = build_command(PY, args.train_script, cmd_args, extra_flags)
        env = os.environ.copy()
        if cuda_devices:
            dev = cuda_devices[job_idx % len(cuda_devices)]
            env['CUDA_VISIBLE_DEVICES'] = str(dev)
        print(f"[RUN {run_id}] CMD: {' '.join(cmd)}")
        if args.dry_run:
            return {
                'job': job, 'started_at': started_at, 'finished_at': None,
                'train_returncode': None, 'train_summary': None, 'eval_metrics': None
            }
        rc = subprocess.run(cmd, env=env).returncode
        train_summary, train_summary_file = find_train_summary(run_out, variant, seed)
        eval_metrics = None
        eval_file = None
        if args.run_eval:
            ckpt = Path(run_out) / variant / f"seed_{seed}" / f"best_specrnet_{variant}_seed{seed}.pt"
            if ckpt.exists():
                eval_out = Path(run_out) / variant / f"seed_{seed}" / "eval"
                eval_out.mkdir(parents=True, exist_ok=True)
                eval_cmd = [PY, args.evaluate_script, '--checkpoint', str(ckpt),
                            '--metadata', args.metadata, '--features', features_path,
                            '--split', args.eval_split, '--output_dir', str(eval_out),
                            '--variant', variant, '--seed', str(seed)]
                print(f"[EVAL {run_id}] CMD: {' '.join(eval_cmd)}")
                subprocess.run(eval_cmd, env=env)
                eval_metrics, eval_file = find_eval_metrics(run_out, variant, seed)

        finished_at = datetime.utcnow().isoformat()
        return {
            'job': job,
            'started_at': started_at,
            'finished_at': finished_at,
            'train_returncode': rc,
            'train_summary': (train_summary or {}),
            'train_summary_file': train_summary_file,
            'eval_metrics': (eval_metrics or {}),
            'eval_file': eval_file
        }

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        for idx, job in enumerate(jobs):
            existing_summary = Path(out_base / job['run_id']) / job['variant'] / f"seed_{job['seed']}" / "summary.json"
            if existing_summary.exists() and not args.overwrite:
                print(f"Skipping existing run (summary present): {job['run_id']}")
                s, sf = find_train_summary(str(out_base / job['run_id']), job['variant'], job['seed'])
                e, ef = find_eval_metrics(str(out_base / job['run_id']), job['variant'], job['seed'])
                row = {
                    'run_id': job['run_id'], 'variant': job['variant'], 'seed': job['seed'],
                    'lr': job['lr'], 'batch_size': job['bs'], 'lfcc': job['lfcc'],
                    'train_features': job['features'], 'run_dir': str(out_base / job['run_id']),
                    'train_summary_file': sf, 'best_f1': (s.get('best_f1') if s else None),
                    'precise_ms': (s.get('precise_ms') if s else None), 'params': (s.get('params') if s else None),
                    'train_returncode': None,
                    'eval_metrics_file': ef, 'eval_f1': (e.get('f1') if e else None),
                    'eval_auc': (e.get('auc') if e else None), 'eval_eer': (e.get('eer') if e else None),
                    'eval_accuracy': (e.get('accuracy') if e else None), 'eval_precision': (e.get('precision') if e else None),
                    'eval_recall': (e.get('recall') if e else None),
                    'started_at': None, 'finished_at': None
                }
                append_csv(agg_csv, row, header)
                continue
            futures[ex.submit(worker, idx, job)] = job

        for fut in as_completed(futures):
            res = fut.result()
            job = res['job']
            run_dir = Path(out_base) / job['run_id']
            train_summary = res.get('train_summary') or {}
            eval_metrics = res.get('eval_metrics') or {}
            row = {
                'run_id': job['run_id'], 'variant': job['variant'], 'seed': job['seed'],
                'lr': job['lr'], 'batch_size': job['bs'], 'lfcc': job['lfcc'],
                'train_features': job['features'], 'run_dir': str(run_dir),
                'train_summary_file': res.get('train_summary_file'),
                'best_f1': train_summary.get('best_f1'),
                'precise_ms': train_summary.get('precise_ms'),
                'params': train_summary.get('params'),
                'train_returncode': res.get('train_returncode'),
                'eval_metrics_file': res.get('eval_file'),
                'eval_f1': eval_metrics.get('f1'),
                'eval_auc': eval_metrics.get('auc'),
                'eval_eer': eval_metrics.get('eer'),
                'eval_accuracy': eval_metrics.get('accuracy'),
                'eval_precision': eval_metrics.get('precision'),
                'eval_recall': eval_metrics.get('recall'),
                'started_at': res.get('started_at'),
                'finished_at': res.get('finished_at')
            }
            append_csv(agg_csv, row, header)
            print(f"Completed job {job['run_id']}: best_f1={row['best_f1']} eval_f1={row['eval_f1']}")

    print("All jobs processed. Aggregated results at:", str(agg_csv))

if __name__ == '__main__':
    main()
