import argparse
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fftpack import dct
from tqdm import tqdm

def extract_lfcc_custom(y, sr=16000, n_lfcc=80):
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=400))
    freqs = np.linspace(0, sr / 2, 512 // 2 + 1)
    edges = np.linspace(0, sr / 2, n_lfcc + 2)
    fb = np.zeros((n_lfcc, len(freqs)))
    for i in range(1, n_lfcc + 1):
        l, c, r = edges[i-1], edges[i], edges[i+1]
        fb[i-1] = np.maximum(0, np.minimum((freqs-l)/(c-l), (r-freqs)/(r-c)))
    energies = np.log(np.dot(fb, stft) + 1e-6)
    return dct(energies, type=2, axis=0, norm='ortho')[:n_lfcc].astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting LFCC"):
        y, _ = librosa.load(row['filepath'], sr=16000, duration=4.0)
        if len(y) < 64000: y = np.pad(y, (0, 64000 - len(y)))
        feat = extract_lfcc_custom(y)
        if feat.shape[1] > 404: feat = feat[:, :404]
        elif feat.shape[1] < 404: feat = np.pad(feat, ((0,0), (0, 404 - feat.shape[1])))
        
        # 修正：唯一文件名标识
        save_name = f"{row['label']}_{row['utt_id']}.npy"
        np.save(Path(args.output_dir) / save_name, feat)

if __name__ == '__main__':
    main()