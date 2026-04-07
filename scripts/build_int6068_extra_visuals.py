#!/usr/bin/env python3
"""Build additional report visuals for INT6068.

Outputs in reports/int6068_final_package:
- training_curves_baseline_3seeds.png
- training_curves_core_variants_seed42.png
- lfcc_real_vs_fake.png
- params_vs_inference_scatter.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "reports" / "int6068_final_package"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_train_log(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_training_curves_baseline() -> None:
    """Plot train loss and val F1 for baseline seeds."""
    run_paths = {
        "default_s42": ROOT / "results_paper_aligned" / "default" / "seed_42" / "train_log.csv",
        "default_s123": ROOT / "results_paper_aligned" / "default" / "seed_123" / "train_log.csv",
        "default_s3407": ROOT / "results_paper_aligned" / "default" / "seed_3407" / "train_log.csv",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, path in run_paths.items():
        df = _read_train_log(path)
        if df is None:
            continue
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.5, label=name)
        axes[1].plot(df["epoch"], df["val_f1"], marker="o", linewidth=1.5, label=name)

    axes[0].set_title("Baseline Train Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Baseline Val F1 vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val F1")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves_baseline_3seeds.png", dpi=180)
    plt.close(fig)


def plot_training_curves_core_variants() -> None:
    """Plot train loss and val F1 for core variants at seed 42."""
    run_paths = {
        "default_4s": ROOT / "results_paper_aligned" / "default" / "seed_42" / "train_log.csv",
        "no-att_4s": ROOT / "results_extensions" / "no-att" / "seed_42" / "train_log.csv",
        "gap_4s": ROOT / "results_extensions" / "gap" / "seed_42" / "train_log.csv",
        "default_1s": ROOT / "results_duration" / "default" / "seed_42" / "train_log.csv",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, path in run_paths.items():
        df = _read_train_log(path)
        if df is None:
            continue
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.5, label=name)
        axes[1].plot(df["epoch"], df["val_f1"], marker="o", linewidth=1.5, label=name)

    axes[0].set_title("Core Experiments Train Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Core Experiments Val F1 vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val F1")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves_core_variants_seed42.png", dpi=180)
    plt.close(fig)


def _iter_feature_candidates(metadata: pd.DataFrame, label: str, feature_dir: Path) -> Iterable[Tuple[Path, str]]:
    subset = metadata[metadata["label"] == label]
    for _, row in subset.iterrows():
        utt_id = str(row["utt_id"])
        file_name = f"{row['generator']}_{row['label']}_{utt_id}.npy"
        feature_path = feature_dir / file_name
        yield feature_path, utt_id


def plot_lfcc_real_vs_fake() -> None:
    """Visualize one real and one fake LFCC sample and their absolute difference."""
    metadata_path = ROOT / "metadata_kaggle.csv"
    feature_dir = ROOT / "lfcc_4s"

    if not metadata_path.exists() or not feature_dir.exists():
        return

    metadata = pd.read_csv(metadata_path)

    real_feat = None
    real_id = ""
    for path, utt_id in _iter_feature_candidates(metadata, "real", feature_dir):
        if path.exists():
            real_feat = np.load(path)
            real_id = utt_id
            break

    fake_feat = None
    fake_id = ""
    for path, utt_id in _iter_feature_candidates(metadata, "fake", feature_dir):
        if path.exists():
            fake_feat = np.load(path)
            fake_id = utt_id
            break

    if real_feat is None or fake_feat is None:
        return

    # Make the difference map interpretable even if amplitudes vary.
    diff = np.abs(fake_feat - real_feat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(real_feat, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Real LFCC\n{real_id}")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("LFCC Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fake_feat, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title(f"Fake LFCC\n{fake_id}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("LFCC Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("|Fake - Real| LFCC")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("LFCC Bin")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "lfcc_real_vs_fake.png", dpi=180)
    plt.close(fig)


def plot_params_vs_inference() -> None:
    """Plot parameter count against inference time for major experiment points."""
    summary_points: Dict[str, Path] = {
        "default": ROOT / "results_paper_aligned" / "default" / "seed_42" / "summary.json",
        "no-att": ROOT / "results_extensions" / "no-att" / "seed_42" / "summary.json",
        "gap": ROOT / "results_extensions" / "gap" / "seed_42" / "summary.json",
        "duration_1s": ROOT / "results_duration" / "default" / "seed_42" / "summary.json",
        "weighted_on": ROOT / "results_weighted" / "default" / "seed_42" / "summary.json",
        "weighted_off": ROOT / "results_weighted_control" / "default" / "seed_42" / "summary.json",
    }

    labels = []
    params = []
    ms = []

    for label, path in summary_points.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        labels.append(label)
        params.append(float(data["params"]))
        ms.append(float(data["precise_ms"]))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(params, ms, s=120, c="#2A9D8F", alpha=0.9)

    for x, y, label in zip(params, ms, labels):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_title("Params vs Inference Time")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Inference Time (ms/sample)")
    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "params_vs_inference_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_training_curves_baseline()
    plot_training_curves_core_variants()
    plot_lfcc_real_vs_fake()
    plot_params_vs_inference()
    print(f"Extra visuals generated in: {OUT_DIR}")


if __name__ == "__main__":
    main()
