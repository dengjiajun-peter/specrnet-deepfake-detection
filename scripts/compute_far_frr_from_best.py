#!/usr/bin/env python3
"""
Find best runs for specified variants and compute FAR/FRR on test split.
Searches for ablation_results.csv files, selects best run by eval_f1, locates checkpoint,
and runs inference on metadata_kaggle.csv using the run's features dir.
Writes results to reports/csv/far_frr_summary.csv and prints details.
"""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import argparse

REPO = Path('.').resolve()
REPORTS = REPO / 'reports' / 'csv'
REPORTS.mkdir(parents=True, exist_ok=True)

from model import SpecRNet
from dataset import LFCCDataset
from torch.utils.data import DataLoader


def find_all_ablation_csvs(root: Path):
    files = list(root.glob('**/ablation_results.csv'))
    # include task root results if outside
    extra = []
    task_root = Path(os.environ.get('TASK_RESULTS_ROOT', 'results_task_ablation'))
    if task_root.exists():
        extra += list(task_root.glob('**/ablation_results.csv'))
    uniq = {}
    for p in files + extra:
        try:
            rp = str(p.resolve())
            uniq[rp] = p
        except Exception:
            continue
    return list(uniq.values())


def pick_best_run(variant_name, csv_paths):
    best = None
    best_score = -1.0
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if 'variant' not in df.columns:
            continue
        # normalize variant strings
        matches = df[df['variant'].astype(str).str.lower() == variant_name.lower()]
        for _, r in matches.iterrows():
            try:
                score = float(r.get('eval_f1', float('nan')))
            except Exception:
                score = float('nan')
            if pd.isna(score):
                continue
            if score > best_score:
                best_score = score
                best = (p, r.to_dict())
    return best


def locate_checkpoint(run_row):
    # run_dir may have backslashes; use Path
    run_dir = Path(run_row.get('run_dir', ''))
    variant = run_row.get('variant')
    seed = run_row.get('seed')
    # candidate path pattern
    candidates = []
    if run_dir and run_dir.exists():
        # look for best_specrnet files under run_dir
        candidates += list(run_dir.glob('**/best_specrnet*.pt'))
        # also try run_dir/<variant>/seed_<seed>/best_*.pt
        try:
            cand = run_dir / variant / f"seed_{seed}"
            if cand.exists():
                candidates += list(cand.glob('best_specrnet*.pt'))
        except Exception:
            pass
    # fallback: search repo for best_specrnet_{variant}
    candidates += list(REPO.glob(f'**/best_specrnet*{variant}*.pt'))
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        # try any best_specrnet
        candidates = [c for c in REPO.glob('**/best_specrnet*.pt') if c.exists()]
    if not candidates:
        return None
    # choose largest file (heuristic)
    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        conf = ckpt.get('config', None)
        variant = ckpt.get('variant', None)
    else:
        state = ckpt
        conf = None
        variant = None
    return state, conf, variant


def compute_far_frr_for_checkpoint(ckpt_path, metadata, features, variant=None, seed=None, device='cpu'):
    state, conf, ckpt_variant = load_checkpoint(ckpt_path, device)
    # conf fallback via config.py
    if conf is None:
        try:
            import config as repo_config
            conf = repo_config.get_specrnet_config(input_channels=1, variant=variant, seed=seed)
        except Exception:
            conf = {'filts': [1, [1, 20], [20, 64]], 'gru_node': 64, 'nb_gru_layer': 2, 'nb_fc_node': 64, 'nb_classes': 2}
    model_variant = ckpt_variant or variant or 'default'
    model = SpecRNet(conf, variant=model_variant).to(device)
    # try to load state dict robustly
    try:
        model.load_state_dict(state)
    except Exception:
        # if state looks like a raw state_dict with "module." prefixes, strip
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state)

    ds = LFCCDataset(metadata, features, 'test')
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    # cm format: [[tn, fp],[fn, tp]] where label 0=real,1=fake
    tn, fp, fn, tp = cm.ravel()
    total_real = tn + fp
    total_fake = fn + tp
    far = float(fn / total_fake) if total_fake > 0 else float('nan')
    frr = float(fp / total_real) if total_real > 0 else float('nan')
    return {
        'checkpoint': str(ckpt_path),
        'variant': model_variant,
        'seed': seed,
        'tn': int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
        'total_real': int(total_real),'total_fake':int(total_fake),'FAR':far,'FRR':frr
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=str, default='no-att,default', help='Comma list of variants to evaluate')
    parser.add_argument('--metadata', type=str, default='metadata_kaggle.csv')
    parser.add_argument('--prefer_features', type=str, default=None, help='Prefer this features dir if run row missing')
    args = parser.parse_args()

    csvs = find_all_ablation_csvs(REPO)
    if not csvs:
        print('No ablation_results.csv found in repo or task root.')
        return

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    results = []
    for v in variants:
        best = pick_best_run(v, csvs)
        if not best:
            print(f'No runs found for variant {v}')
            continue
        csv_path, run_row = best
        print(f"Best run for {v} found in {csv_path}: run_id={run_row.get('run_id')}, eval_f1={run_row.get('eval_f1')}")
        ckpt = locate_checkpoint(run_row)
        if ckpt is None:
            print(f"  [WARN] No checkpoint found for run {run_row.get('run_id')}")
            continue
        # determine feature dir
        features = run_row.get('train_features') or args.prefer_features or 'lfcc_features'
        # compute
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        res = compute_far_frr_for_checkpoint(ckpt, args.metadata, features, variant=run_row.get('variant'), seed=run_row.get('seed'), device=device)
        results.append(res)
        print(json.dumps(res, indent=2))

    if results:
        out = REPORTS / 'far_frr_summary.csv'
        pd.DataFrame(results).to_csv(out, index=False)
        print('\nWrote FAR/FRR summary to', out)

if __name__ == '__main__':
    import argparse
    main()
