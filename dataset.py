import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class LFCCDataset(Dataset):
    def __init__(self, metadata_csv, feature_dir, split):
        df = pd.read_csv(metadata_csv)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 修正：读取唯一文件名（包含 generator）
        feat_path = self.feature_dir / f"{row['generator']}_{row['label']}_{row['utt_id']}.npy"
        try:
            feat = np.load(feat_path)
        except Exception as e:
            print(f"ERROR loading feature file: {feat_path} -> {e}")
            raise

        # validate shape
        if feat.size == 0 or feat.shape != (80, 404):
            print(f"CORRUPT feature shape for {feat_path}: {getattr(feat, 'shape', None)} size={getattr(feat, 'size', None)}")
            raise ValueError(f"Invalid feature shape: {feat_path} -> {getattr(feat, 'shape', None)}")

        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return torch.from_numpy(feat).unsqueeze(0), torch.tensor(row['label_id']), row['utt_id']