import argparse
import os
import sys
import time
import json
import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Ensure local project path is importable regardless launch cwd.
REPO_PATH = Path(__file__).resolve().parent
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from model import SpecRNet
try:
    import config as repo_config
    def get_config(variant: str = 'default', seed: int = None):
        conf = repo_config.get_specrnet_config(input_channels=1, variant=variant, seed=seed)
        conf['nb_classes'] = 2
        return conf
except Exception as e:
    print(f"Warning: Could not load repo config ({e}), using fallback.")
    def get_config(variant: str = 'default', seed: int = None):
        conf = {
            'filts': [1, [1, 20], [20, 64]],
            'gru_node': 64,
            'nb_gru_layer': 2,
            'nb_fc_node': 64,
            'nb_classes': 2,
            'use_attention': True,
            'head': 'gru',
            'variant': variant,
            'seed': seed,
        }
        if variant in ('gru1','1layer'):
            conf['nb_gru_layer'] = 1
        if variant in ('no-att','no_attention'):
            conf['use_attention'] = False
        if variant == 'gap':
            conf['use_attention'] = False
            conf['head'] = 'gap'
        if variant == 'no-gru':
            conf['head'] = 'last'
        return conf

from dataset import LFCCDataset


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


try:
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    def compute_eer(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # use linear interpolation of tpr over fpr
        func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(tpr[0], tpr[-1]))
        try:
            eer = brentq(lambda x: 1.0 - x - func(x), 0.0, 1.0)
            return float(eer)
        except Exception:
            # fallback to approximate method
            fnr = 1.0 - tpr
            idx = np.nanargmin(np.abs(fnr - fpr))
            return float((fpr[idx] + fnr[idx]) / 2.0)
except Exception:
    # scipy not available, use fallback
    from sklearn.metrics import roc_curve

    def compute_eer(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        return float((fpr[idx] + fnr[idx]) / 2.0)


def validate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        'f1': float(f1_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_prob)),
        'eer': float(compute_eer(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
    }
    return metrics


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    conf = get_config(variant=args.variant, seed=args.seed)
    model = SpecRNet(conf, variant=args.variant, device=device).to(device)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params_count:,}")

    # datasets & loaders
    train_ds = LFCCDataset(args.metadata, args.features, 'train')
    val_ds = LFCCDataset(args.metadata, args.features, 'val')

    def worker_init_fn(worker_id):
        if args.seed is not None:
            np.random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # optimizer / scheduler / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # weighted loss (CrossEntropy with class weights)
    if args.weighted_loss:
        try:
            counts = train_ds.df['label_id'].value_counts().to_dict()
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            total = n0 + n1
            w0 = total / (n0 + 1e-6)
            w1 = total / (n1 + 1e-6)
            weight = torch.tensor([w0, w1], dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        except Exception:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # prepare results path & csv
    out_dir = Path(args.out_dir) / args.variant / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'train_log.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow([
            'epoch', 'train_loss', 'val_f1', 'val_auc', 'val_eer',
            'val_acc', 'val_precision', 'val_recall', 'inf_ms', 'grad_norm'
        ])

    # Start from -inf so we always save at least one best checkpoint.
    best_f1 = float('-inf')
    best_epoch = -1

    history = {'loss': [], 'val_f1': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            optimizer.step()
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / len(train_loader)
        v_metrics = validate(model, val_loader, device)

        # inference timing (quick single-batch measure)
        model.eval()
        dummy = torch.randn(1, 1, 80, 404).to(device)
        with torch.no_grad():
            for _ in range(5): _ = model(dummy)
            if device == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20): _ = model(dummy)
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = (time.time() - t0) / 20.0
            inf_ms = elapsed * 1000.0

        history['loss'].append(train_loss)
        history['val_f1'].append(v_metrics['f1'])

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val F1: {v_metrics['f1']:.4f} | Val AUC: {v_metrics['auc']:.4f} | Val EER: {v_metrics['eer']:.4f}")

        # append CSV
        with open(csv_path, 'a', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow([
                epoch,
                train_loss,
                v_metrics['f1'],
                v_metrics['auc'],
                v_metrics['eer'],
                v_metrics['accuracy'],
                v_metrics['precision'],
                v_metrics['recall'],
                f"{inf_ms:.3f}",
                f"{total_norm:.6f}",
            ])

        # checkpoint by best val f1
        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            best_epoch = epoch
            ckpt = {
                'state_dict': model.state_dict(),
                'config': conf,
                'variant': args.variant,
                'seed': args.seed,
                'epoch': epoch,
                'best_f1': best_f1,
            }
            torch.save(ckpt, out_dir / f"best_specrnet_{args.variant}_seed{args.seed}.pt")

    # save history
    with open(out_dir / 'history.json', 'w') as hf:
        json.dump(history, hf, indent=2)

    # final precise inference benchmark
    model.eval()
    dummy = torch.randn(1, 1, 80, 404).to(device)
    with torch.no_grad():
        for _ in range(20): _ = model(dummy)
        if device == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(100): _ = model(dummy)
        if device == 'cuda': torch.cuda.synchronize()
        precise_ms = (time.time() - start) * 10.0
        print(f"Precise Inference Time: {precise_ms:.2f} ms / sample")
    # write final precision to summary
    with open(out_dir / 'summary.json', 'w') as sf:
        json.dump(
            {'params': params_count, 'precise_ms': precise_ms, 'best_f1': best_f1, 'best_epoch': best_epoch},
            sf,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='default', choices=['default','no-att','gru1','gap','no-gru'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metadata', type=str, default='metadata_final.csv')
    parser.add_argument('--features', type=str, default='lfcc_features')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none','cosine'])
    parser.add_argument('--weighted_loss', action='store_true')
    args = parser.parse_args()
    # keep args.epochs aligned with parser
    args.epochs = args.epochs
    train(args)


if __name__ == '__main__':
    main()