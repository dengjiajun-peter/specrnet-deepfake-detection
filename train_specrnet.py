import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
from pathlib import Path
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 导入配置和模型
REPRO_PATH = "./specrnet" 
sys.path.insert(0, REPRO_PATH)
from model import SpecRNet
try:
    import config as repo_config
    def get_config():
        conf = repo_config.get_specrnet_config(input_channels=1)
        # 强制覆盖，确保是 2 分类，防止 CUDA assert 错误
        conf['nb_classes'] = 2 
        return conf
except Exception as e:
    print(f"Warning: Could not load repo config ({e}), using fallback.")
    def get_config():
        return {
            'filts': [1, [1, 20], [20, 64]], 
            'gru_node': 64, 
            'nb_gru_layer': 2, 
            'nb_fc_node': 64, 
            'nb_classes': 2
        }
from dataset import LFCCDataset

def validate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_p.extend(preds.cpu().numpy())
            all_y.extend(y.cpu().numpy())
    
    # --- 关键：确保这行在 for 循环结束之后执行 ---
    return f1_score(all_y, all_p)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 正确写法：
    model = SpecRNet(get_config()).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_ds = LFCCDataset('metadata_final.csv', 'lfcc_features', 'train')
    val_ds = LFCCDataset('metadata_final.csv', 'lfcc_features', 'val')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    history = {'loss': [], 'val_f1': []}

    for epoch in range(15):
        model.train()
        total_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        v_f1 = validate(model, val_loader, device)
        history['loss'].append(total_loss/len(train_loader))
        history['val_f1'].append(v_f1)
        print(f"Epoch {epoch} | Loss: {history['loss'][-1]:.4f} | Val F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), "best_specrnet.pt")

    # 学术 Benchmark：精确推理时间测定
    model.eval()
    dummy = torch.randn(1, 1, 80, 404).to(device)
    with torch.no_grad():
        for _ in range(20): _ = model(dummy) # Warmup
        if device == "cuda": torch.cuda.synchronize()
        start = time.time()
        for _ in range(100): _ = model(dummy)
        if device == "cuda": torch.cuda.synchronize()
        print(f"Precise Inference Time: {(time.time()-start)*10:.2f} ms / sample")

if __name__ == '__main__':
    train()