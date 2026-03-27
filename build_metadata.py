import argparse
import random
import re
from pathlib import Path

import pandas as pd


def extract_id(path: Path):
    """
    从文件名中提取 LJ001-0001 这种 utterance id
    适配：
    - LJ001-0001.wav
    - LJ001-0001_generated.wav
    """
    match = re.search(r'(LJ\d{3}-\d{4})', path.stem)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="Build paired metadata for LJSpeech real + WaveFake fake.")
    parser.add_argument("--real_dir", type=str, default="LJSpeech-1.1/wavs",
                        help="Path to real LJSpeech wav directory")
    parser.add_argument("--fake_dir", type=str, default="generated_audio/ljspeech_hifiGAN",
                        help="Path to fake generator directory")
    parser.add_argument("--output_csv", type=str, default="metadata_final.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real directory not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake directory not found: {fake_dir}")

    # 从 fake_dir 名字里提取 generator 名称
    generator_name = fake_dir.name

    print("Scanning real audio...")
    real_files = {}
    for f in real_dir.glob("*.wav"):
        uid = extract_id(f)
        if uid:
            real_files[uid] = str(f.resolve())

    print("Scanning fake audio...")
    fake_files = {}
    for f in fake_dir.glob("*.wav"):
        uid = extract_id(f)
        if uid:
            fake_files[uid] = str(f.resolve())

    common_ids = sorted(set(real_files.keys()) & set(fake_files.keys()))
    if not common_ids:
        raise RuntimeError("No paired utterance IDs found between real and fake directories.")

    print(f"Paired utterances found: {len(common_ids)}")

    random.seed(args.seed)
    random.shuffle(common_ids)

    n = len(common_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_ids = set(common_ids[:train_end])
    val_ids = set(common_ids[train_end:val_end])
    test_ids = set(common_ids[val_end:])

    rows = []
    for uid in common_ids:
        if uid in train_ids:
            split = "train"
        elif uid in val_ids:
            split = "val"
        else:
            split = "test"

        rows.append({
            "filepath": real_files[uid],
            "label": "real",
            "label_id": 0,
            "utt_id": uid,
            "split": split,
            "generator": "real"
        })

        rows.append({
            "filepath": fake_files[uid],
            "label": "fake",
            "label_id": 1,
            "utt_id": uid,
            "split": split,
            "generator": generator_name
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved: {args.output_csv}")
    print("\nSplit summary:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    print("\nExample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
