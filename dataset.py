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
        # 修正：读取唯一文件名
        feat_path = self.feature_dir / f"{row['label']}_{row['utt_id']}.npy"
        feat = np.load(feat_path)
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return torch.from_numpy(feat).unsqueeze(0), torch.tensor(row['label_id']), row['utt_id']