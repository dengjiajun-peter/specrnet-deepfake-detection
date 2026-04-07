#!/usr/bin/env python3
"""Build final INT6068 result package from existing experiment outputs.

Outputs under reports/int6068_final_package:
- baseline_table.csv
- baseline_mean_std.csv
- extension_table.csv
- speed_complexity_table.csv
- ablation_bar_auc_eer.png
- duration_vs_performance.png
- roc_overview.png
- confusion_overview.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_image_panel(image_items: List[Dict], out_path: Path, title: str) -> None:
    valid = [x for x in image_items if Path(x["path"]).exists()]
    if not valid:
        return

    cols = len(valid)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4.5))
    if cols == 1:
        axes = [axes]

    for ax, item in zip(axes, valid):
        img = mpimg.imread(item["path"])
        ax.imshow(img)
        ax.set_title(item["title"])
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(".")
    eval_root = root / "results_eval"
    out_dir = root / "reports" / "int6068_final_package"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: strict paper-aligned track.
    baseline_runs = [
        ("default", 42, eval_root / "default_s42" / "metrics.json"),
        ("default", 123, eval_root / "default_s123" / "metrics.json"),
        ("default", 3407, eval_root / "default_s3407" / "metrics.json"),
    ]

    baseline_rows = []
    acc_vals, f1_vals, auc_vals, eer_vals = [], [], [], []

    for variant, seed, path in baseline_runs:
        m = load_json(path)
        row = {
            "variant": variant,
            "seed": seed,
            "AUC": m["auc"],
            "EER": m["eer"],
            "F1": m["f1"],
            "Accuracy": m["accuracy"],
        }
        baseline_rows.append(row)
        acc_vals.append(m["accuracy"])
        f1_vals.append(m["f1"])
        auc_vals.append(m["auc"])
        eer_vals.append(m["eer"])

    write_csv(
        out_dir / "baseline_table.csv",
        baseline_rows,
        ["variant", "seed", "AUC", "EER", "F1", "Accuracy"],
    )

    baseline_mean_std = [
        {
            "metric": "Accuracy",
            "mean": mean(acc_vals),
            "std": safe_std(acc_vals),
        },
        {
            "metric": "F1",
            "mean": mean(f1_vals),
            "std": safe_std(f1_vals),
        },
        {
            "metric": "AUC",
            "mean": mean(auc_vals),
            "std": safe_std(auc_vals),
        },
        {
            "metric": "EER",
            "mean": mean(eer_vals),
            "std": safe_std(eer_vals),
        },
    ]

    write_csv(
        out_dir / "baseline_mean_std.csv",
        baseline_mean_std,
        ["metric", "mean", "std"],
    )

    # Extensions: one-seed comparisons.
    extension_map = {
        "no-att": eval_root / "noatt_s42" / "metrics.json",
        "gap": eval_root / "gap_s42" / "metrics.json",
        "duration_1s": eval_root / "duration1s_s42" / "metrics.json",
        "weighted_loss_off": eval_root / "weighted_control_s42" / "metrics.json",
        "weighted_loss_on": eval_root / "weighted_on_s42" / "metrics.json",
    }

    extension_rows = []
    for name, path in extension_map.items():
        m = load_json(path)
        extension_rows.append(
            {
                "experiment": name,
                "AUC": m["auc"],
                "EER": m["eer"],
                "F1": m["f1"],
                "Accuracy": m["accuracy"],
                "interpretation": "fill_in_report_text",
            }
        )

    write_csv(
        out_dir / "extension_table.csv",
        extension_rows,
        ["experiment", "AUC", "EER", "F1", "Accuracy", "interpretation"],
    )

    # Speed/complexity table from summary.json.
    speed_items = [
        ("default", root / "results_paper_aligned" / "default" / "seed_42" / "summary.json", "paper-aligned baseline"),
        ("no-att", root / "results_extensions" / "no-att" / "seed_42" / "summary.json", "attention removed"),
        ("gap", root / "results_extensions" / "gap" / "seed_42" / "summary.json", "temporal head replaced by GAP"),
        ("duration_1s_default", root / "results_duration" / "default" / "seed_42" / "summary.json", "short-utterance condition"),
        ("weighted_loss_on", root / "results_weighted" / "default" / "seed_42" / "summary.json", "class-weighted CE"),
        ("weighted_loss_off", root / "results_weighted_control" / "default" / "seed_42" / "summary.json", "control"),
    ]

    speed_rows = []
    for variant, path, note in speed_items:
        s = load_json(path)
        speed_rows.append(
            {
                "variant": variant,
                "params": s.get("params"),
                "inference_ms_per_sample": s.get("precise_ms"),
                "notes": note,
            }
        )

    write_csv(
        out_dir / "speed_complexity_table.csv",
        speed_rows,
        ["variant", "params", "inference_ms_per_sample", "notes"],
    )

    # Figure: ablation bars (AUC + EER for default/no-att/gap).
    names = ["default", "no-att", "gap"]
    aucs = [
        load_json(eval_root / "default_s42" / "metrics.json")["auc"],
        load_json(eval_root / "noatt_s42" / "metrics.json")["auc"],
        load_json(eval_root / "gap_s42" / "metrics.json")["auc"],
    ]
    eers = [
        load_json(eval_root / "default_s42" / "metrics.json")["eer"],
        load_json(eval_root / "noatt_s42" / "metrics.json")["eer"],
        load_json(eval_root / "gap_s42" / "metrics.json")["eer"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, aucs, color="#2A9D8F")
    axes[0].set_title("Ablation Test AUC")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(names, eers, color="#E76F51")
    axes[1].set_title("Ablation Test EER")
    axes[1].set_ylim(0.0, max(eers) * 1.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ablation_bar_auc_eer.png", dpi=180)
    plt.close(fig)

    # Figure: duration vs performance.
    m4 = load_json(eval_root / "default_s42" / "metrics.json")
    m1 = load_json(eval_root / "duration1s_s42" / "metrics.json")

    x = ["4s", "1s"]
    f1_vals_dur = [m4["f1"], m1["f1"]]
    auc_vals_dur = [m4["auc"], m1["auc"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(x, f1_vals_dur, marker="o", color="#264653")
    axes[0].set_title("Duration vs F1")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(linestyle="--", alpha=0.3)

    axes[1].plot(x, auc_vals_dur, marker="o", color="#F4A261")
    axes[1].set_title("Duration vs AUC")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "duration_vs_performance.png", dpi=180)
    plt.close(fig)

    # Figure: ROC and confusion overviews.
    build_image_panel(
        [
            {"title": "default", "path": str(eval_root / "default_s42" / "roc_curve.png")},
            {"title": "no-att", "path": str(eval_root / "noatt_s42" / "roc_curve.png")},
            {"title": "gap", "path": str(eval_root / "gap_s42" / "roc_curve.png")},
        ],
        out_dir / "roc_overview.png",
        "ROC Curves (Baseline + Core Ablations)",
    )

    build_image_panel(
        [
            {"title": "weighted_off", "path": str(eval_root / "weighted_control_s42" / "confusion_matrix.png")},
            {"title": "weighted_on", "path": str(eval_root / "weighted_on_s42" / "confusion_matrix.png")},
        ],
        out_dir / "confusion_overview.png",
        "Confusion Matrix (Weighted Loss Comparison)",
    )

    # Store leakage note in the package for traceability.
    leak = load_json(root / "reports" / "leakage_audit_tonight" / "leakage_summary.json")
    (out_dir / "leakage_summary_copy.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    print(f"Final package generated: {out_dir}")


if __name__ == "__main__":
    main()
