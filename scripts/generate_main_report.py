#!/usr/bin/env python3
"""Generate a compact experiment report from existing training/evaluation outputs.

Artifacts produced:
- main_results_table.csv
- ablation_metrics.png (AUC / F1 / EER bars)
- roc_overview.png (panel of existing per-variant ROC images)
- confusion_overview.png (panel of existing per-variant confusion matrices)

This script consumes existing files under results directory and does not retrain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else float("nan")


def std_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def load_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_variants(results_dir: Path) -> List[str]:
    variants = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("eval_test"):
            continue
        # A valid variant directory should contain at least one seed_* run directory.
        if any(child.is_dir() and child.name.startswith("seed_") for child in p.iterdir()):
            variants.append(p.name)
    return sorted(variants)


def collect_training_stats(results_dir: Path, variant: str) -> Dict:
    variant_dir = results_dir / variant
    seed_dirs = sorted([p for p in variant_dir.glob("seed_*") if p.is_dir()])

    best_f1_values = []
    params_values = []
    infer_values = []

    for sd in seed_dirs:
        summary = load_json_if_exists(sd / "summary.json")
        if not summary:
            continue
        best_f1_values.append(summary.get("best_f1"))
        params_values.append(summary.get("params"))
        infer_values.append(summary.get("precise_ms"))

    return {
        "runs": len(seed_dirs),
        "val_best_f1_mean": mean_or_nan(best_f1_values),
        "val_best_f1_std": std_or_nan(best_f1_values),
        "params_mean": mean_or_nan(params_values),
        "inference_ms_mean": mean_or_nan(infer_values),
    }


def collect_eval_stats(results_dir: Path, variant: str) -> Dict:
    # Preferred layout: results_dir/eval_test_<variant>/metrics.json
    metrics_path = results_dir / f"eval_test_{variant}" / "metrics.json"

    # Backward compatibility: some runs may have a shared eval_test folder.
    if not metrics_path.exists() and variant == "default":
        fallback = results_dir / "eval_test" / "metrics.json"
        if fallback.exists():
            metrics_path = fallback

    metrics = load_json_if_exists(metrics_path) or {}
    return {
        "test_accuracy": metrics.get("accuracy", np.nan),
        "test_precision": metrics.get("precision", np.nan),
        "test_recall": metrics.get("recall", np.nan),
        "test_f1": metrics.get("f1", np.nan),
        "test_auc": metrics.get("auc", np.nan),
        "test_eer": metrics.get("eer", np.nan),
        "eval_dir": str(metrics_path.parent) if metrics_path.exists() else "",
    }


def save_ablation_plot(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df = plot_df.sort_values("variant")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(plot_df["variant"], plot_df["test_auc"], color="#2A9D8F")
    axes[0].set_title("Test AUC")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(plot_df["variant"], plot_df["test_f1"], color="#E9C46A")
    axes[1].set_title("Test F1")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(plot_df["variant"], plot_df["test_eer"], color="#E76F51")
    axes[2].set_title("Test EER (lower is better)")
    axes[2].set_ylim(0.0, max(0.1, float(np.nanmax(plot_df["test_eer"])) * 1.2 if not np.isnan(np.nanmax(plot_df["test_eer"])) else 0.1))

    for ax in axes:
        ax.set_xlabel("Variant")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_image_overview(df: pd.DataFrame, image_name: str, out_path: Path, title: str) -> None:
    rows = []
    for _, r in df.iterrows():
        eval_dir = r.get("eval_dir", "")
        if not eval_dir:
            continue
        candidate = Path(eval_dir) / image_name
        if candidate.exists():
            rows.append((r["variant"], candidate))

    if not rows:
        return

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (variant, img_path) in zip(axes, rows):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(str(variant))
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate table and plots from ablation results.")
    parser.add_argument("--results_dir", type=str, default="results_kaggle_balanced")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "report_assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = find_variants(results_dir)
    if not variants:
        raise RuntimeError(f"No variant directories found under: {results_dir}")

    all_rows = []
    for variant in variants:
        row = {"variant": variant}
        row.update(collect_training_stats(results_dir, variant))
        row.update(collect_eval_stats(results_dir, variant))
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "main_results_table.csv", index=False)

    save_ablation_plot(df, out_dir / "ablation_metrics.png")
    save_image_overview(df, "roc_curve.png", out_dir / "roc_overview.png", "ROC Curves by Variant")
    save_image_overview(
        df,
        "confusion_matrix.png",
        out_dir / "confusion_overview.png",
        "Confusion Matrices by Variant",
    )

    print(f"Report generated at: {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
