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
    parser.add_argument('--duration', type=float, default=4.0, help='Audio duration in seconds to load (e.g., 1.0 or 4.0)')
    parser.add_argument('--n_lfcc', type=int, default=80, help='Number of LFCC coefficients')
    parser.add_argument('--n_frames', type=int, default=404, help='Number of time frames to pad/trim to')
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate')
    parser.add_argument('--dry_run', action='store_true', help='Print planned saves but do not write .npy files')
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    min_samples = int(args.duration * args.sr)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting LFCC"):
        y, _ = librosa.load(row['filepath'], sr=args.sr, duration=args.duration)
        # Use wrap padding to avoid silent-zero artifacts that leak split info
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode='wrap')
        feat = extract_lfcc_custom(y, sr=args.sr, n_lfcc=args.n_lfcc)
        if feat.shape[1] > args.n_frames:
            feat = feat[:, :args.n_frames]
        elif feat.shape[1] < args.n_frames:
            # pad feature time axis using edge to avoid introducing zeros
            feat = np.pad(feat, ((0,0), (0, args.n_frames - feat.shape[1])), mode='edge')

        # Unique filename include generator to prevent collisions
        save_name = f"{row['generator']}_{row['label']}_{row['utt_id']}.npy"
        out_path = Path(args.output_dir) / save_name
        if out_path.exists():
            # skip existing (resume support)
            continue
        if args.dry_run:
            print(f"[DRY] Would save: {out_path} shape={feat.shape}")
            continue
        try:
            np.save(out_path, feat)
        except Exception as e:
            print(f"Warning: failed to save {out_path}: {e}")
            # continue processing remaining files
            continue

if __name__ == '__main__':
    main()