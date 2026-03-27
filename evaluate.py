import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader

# 确保能搜到你的 specrnet 文件夹
REPRO_PATH = "./" 
sys.path.insert(0, REPRO_PATH)
from model import SpecRNet
from dataset import LFCCDataset

def get_config():
    return {'filts': [1, [1, 20], [20, 64]], 'gru_node': 64, 'nb_gru_layer': 2, 'nb_fc_node': 64, 'nb_classes': 2}

@torch.no_grad()
def run_evaluation(model, loader, device, output_dir):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    print("Running Inference on FULL Dataset...")
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob))
    }

    # 绘图逻辑
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_specrnet.pt")
    parser.add_argument("--metadata", type=str, default="metadata_final.csv") # 新增
    parser.add_argument("--features", type=str, default="lfcc_features")      # 新增
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    model = SpecRNet(get_config()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 这里的参数现在是动态的了！
    ds = LFCCDataset(args.metadata, args.features, args.split)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    metrics = run_evaluation(model, loader, device, out_path)
    
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n=== Test Results on {args.metadata} ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

if __name__ == "__main__":
    main()