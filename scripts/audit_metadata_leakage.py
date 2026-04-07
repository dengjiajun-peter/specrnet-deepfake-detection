#!/usr/bin/env python3
"""Audit split leakage risks in a metadata CSV.

This script checks common leakage patterns for audio deepfake detection:
1. Exact `utt_id` duplicates across train/val/test.
2. Conversion-pair leakage across splits (e.g., biden-to-obama).
3. Source speaker leakage across splits.

Outputs are written as CSV/JSON files for reporting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"filepath", "label", "label_id", "utt_id", "generator", "split"}


def extract_pair_key(utt_id: str) -> str:
    """Remove chunk suffix so chunks from the same original pair map together."""
    match = re.match(r"^(.*)_chunk\d+$", str(utt_id))
    return match.group(1) if match else str(utt_id)


def extract_source_speaker(pair_key: str) -> str:
    """Extract source speaker token from pair key.

    Examples:
    - biden-to-obama -> biden
    - trump-original -> trump
    """
    if "-to-" in pair_key:
        return pair_key.split("-to-", 1)[0]
    if pair_key.endswith("-original"):
        return pair_key.rsplit("-original", 1)[0]
    return pair_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit potential leakage in metadata split design.")
    parser.add_argument("--metadata", type=str, default="metadata_kaggle.csv", help="Path to metadata CSV")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports/leakage_audit",
        help="Output directory for audit artifacts",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 1) Split x label distribution.
    split_label = (
        df.groupby(["split", "label"]).size().reset_index(name="count").sort_values(["split", "label"])
    )
    split_label.to_csv(out_dir / "split_label_counts.csv", index=False)

    # 2) Exact utt_id leakage across splits.
    utt_split_count = df.groupby("utt_id")["split"].nunique().reset_index(name="split_count")
    utt_multi_split = utt_split_count[utt_split_count["split_count"] > 1].copy()
    if not utt_multi_split.empty:
        utt_multi_split = utt_multi_split.merge(
            df.groupby("utt_id")["split"].agg(lambda s: "|".join(sorted(set(s)))).reset_index(name="splits"),
            on="utt_id",
            how="left",
        )
    utt_multi_split.to_csv(out_dir / "duplicate_utt_id_across_splits.csv", index=False)

    # 3) Pair leakage across splits.
    work = df.copy()
    work["pair_key"] = work["utt_id"].map(extract_pair_key)
    pair_split_count = work.groupby("pair_key")["split"].nunique().reset_index(name="split_count")
    pair_multi_split = pair_split_count[pair_split_count["split_count"] > 1].copy()
    if not pair_multi_split.empty:
        pair_multi_split = pair_multi_split.merge(
            work.groupby("pair_key")["split"].agg(lambda s: "|".join(sorted(set(s)))).reset_index(name="splits"),
            on="pair_key",
            how="left",
        )
    pair_multi_split.to_csv(out_dir / "pair_multi_split.csv", index=False)

    # 4) Source speaker leakage across splits.
    work["source_speaker"] = work["pair_key"].map(extract_source_speaker)
    source_split_count = work.groupby("source_speaker")["split"].nunique().reset_index(name="split_count")
    source_multi_split = source_split_count[source_split_count["split_count"] > 1].copy()
    if not source_multi_split.empty:
        source_multi_split = source_multi_split.merge(
            work.groupby("source_speaker")["split"].agg(lambda s: "|".join(sorted(set(s)))).reset_index(name="splits"),
            on="source_speaker",
            how="left",
        )
    source_multi_split.to_csv(out_dir / "source_speaker_multi_split.csv", index=False)

    summary = {
        "metadata": str(metadata_path),
        "total_rows": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "utt_id_multi_split_count": int(len(utt_multi_split)),
        "pair_key_multi_split_count": int(len(pair_multi_split)),
        "source_speaker_multi_split_count": int(len(source_multi_split)),
        "note": (
            "pair/source leakage means related content appears in multiple splits; "
            "this can inflate validation/test scores."
        ),
    }

    with open(out_dir / "leakage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Leakage audit completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
