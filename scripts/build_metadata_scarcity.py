#!/usr/bin/env python3
"""
Create a metadata CSV that uses only a fraction of the original training set (stratified by label).

Example:
  python scripts/build_metadata_scarcity.py --metadata metadata_multi.csv --out_csv metadata_multi_10pct.csv --train_frac 0.1 --seed 42
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--train_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    if 'split' not in df.columns:
        raise RuntimeError('metadata CSV must contain a "split" column with values train/val/test')

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    sampled_parts = []
    # stratified by label
    for label, g in train_df.groupby('label'):
        n = max(1, int(len(g) * args.train_frac))
        sampled = g.sample(n=n, random_state=args.seed)
        sampled_parts.append(sampled)

    sampled_train = pd.concat(sampled_parts, ignore_index=True)
    out_df = pd.concat([sampled_train, val_df, test_df], ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'Wrote {len(out_df)} rows to {args.out_csv} (train reduced from {len(train_df)} to {len(sampled_train)})')


if __name__ == '__main__':
    main()
