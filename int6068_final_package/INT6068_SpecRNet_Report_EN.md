# INT6068 Group Project Report (English)

## Reproducing SpecRNet for Audio DeepFake Detection

## 1. Project Scope

This project reproduces the core idea of the paper **"SpecRNet: Towards Faster and More Accessible Audio DeepFake Detection"** and extends it with practical experiments on a Kaggle-based dataset.

The work is organized into two clearly separated tracks:

1. **Paper-aligned baseline track**
2. **Project extension track**

This separation is important for fair reporting.

## 2. End-to-End Pipeline

The full workflow of this project is:

1. Prepare metadata (`filepath`, `label`, `label_id`, `utt_id`, `generator`, `split`).
2. Extract LFCC features from audio with fixed settings.
3. Load LFCC tensors via `LFCCDataset`.
4. Train SpecRNet variants with controlled hyperparameters.
5. Evaluate on the test split and export metrics and figures.
6. Aggregate tables and report-ready figures.

### 2.1 Feature Extraction

LFCC extraction settings used in this project:

- sampling rate: 16000
- duration: 4.0s (baseline) and 1.0s (short-utterance extension)
- LFCC bins: 80
- target frames: 404
- short audio padding: wrap
- short frame padding: edge

Feature naming rule:

`{generator}_{label}_{utt_id}.npy`

### 2.2 Dataset Loading

`LFCCDataset` behavior:

- Filters by `split` (`train` / `val` / `test`)
- Loads features with the same naming rule as extraction
- Enforces shape `(80, 404)`
- Applies per-sample normalization
- Returns tensor shape `[1, 80, 404]`, plus `label_id` and `utt_id`

### 2.3 Training Logic

Main training script supports:

- variant selection (`default`, `no-att`, `gru1`, `gap`, `no-gru`)
- seed control
- optimizer settings (`lr`, `weight_decay`)
- scheduler control (`none` / `cosine`)
- weighted loss toggle
- checkpoint save for best validation F1
- logging of loss and validation metrics per epoch

## 3. Paper-Aligned Baseline Track

### 3.1 Setup

- variant: `default`
- seeds: `42, 123, 3407`
- epochs: `10`
- batch size: `128`
- lr: `1e-4`
- weight decay: `1e-4`
- scheduler: `none`
- weighted loss: `off`

### 3.2 Test Results

| Variant | Seed | AUC | EER | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| default | 42 | 0.9981 | 0.0215 | 0.9904 | 0.9831 |
| default | 123 | 0.9996 | 0.0138 | 0.9929 | 0.9875 |
| default | 3407 | 0.9975 | 0.0207 | 0.9934 | 0.9884 |

Baseline mean +- std:

- Accuracy: 0.9863 +- 0.0029
- F1: 0.9922 +- 0.0016
- AUC: 0.9984 +- 0.0011
- EER: 0.0187 +- 0.0042

Interpretation:

- Baseline is strong and stable across three seeds.
- This is reasonably close to paper-style training settings.

## 4. Project Extension Track

### 4.1 Extension Experiments

- `no-att` (attention ablation)
- `gap` (temporal head ablation)
- `duration_1s` (short-utterance robustness)
- `weighted_loss` on/off

### 4.2 Test Results (Current Runs)

| Experiment | AUC | EER | F1 | Accuracy |
|---|---:|---:|---:|---:|
| no-att | 0.9991 | 0.0195 | 0.9934 | 0.9884 |
| gap | 0.9970 | 0.0277 | 0.9892 | 0.9813 |
| duration_1s | 0.9244 | 0.1578 | 0.9622 | 0.9322 |
| weighted_loss_off | 0.9981 | 0.0215 | 0.9904 | 0.9831 |
| weighted_loss_on | 0.9972 | 0.0328 | 0.9898 | 0.9822 |

Interpretation:

- `no-att` is very close to baseline, suggesting attention contributes less than expected on this dataset.
- `gap` is much faster and lighter, with moderate quality loss.
- `1s` shows a large performance drop, matching the short-utterance difficulty trend.
- Weighted loss did not improve the current test metrics.

## 5. Speed and Complexity Analysis

| Variant | Params | Inference (ms/sample) | Note |
|---|---:|---:|---|
| default | 278092 | 1.8656 | paper-aligned baseline |
| no-att | 269352 | 1.6267 | attention removed |
| gap | 136744 | 0.7573 | temporal head replaced by GAP |
| duration_1s_default | 278092 | 1.9406 | short-utterance condition |
| weighted_loss_on | 278092 | 2.0315 | class-weighted CE |
| weighted_loss_off | 278092 | 1.9024 | control |

Key message:

- `gap` offers a major efficiency gain (fewer parameters and lower inference time), but with an EER trade-off.

## 6. Figures Generated for Final Report

The following figures are generated and ready to use:

1. `roc_overview.png`: ROC comparison for baseline and core ablations
2. `confusion_overview.png`: weighted loss on/off confusion comparison
3. `ablation_bar_auc_eer.png`: ablation comparison on AUC and EER
4. `duration_vs_performance.png`: 4s vs 1s performance comparison
5. `training_curves_baseline_3seeds.png`: train loss / val F1 over epochs for baseline seeds
6. `training_curves_core_variants_seed42.png`: train loss / val F1 across core variants
7. `lfcc_real_vs_fake.png`: LFCC example visualization (real vs fake)
8. `params_vs_inference_scatter.png`: efficiency view (model size vs runtime)

## 7. Limitations and Risks

### 7.1 Leakage Risk

From leakage audit:

- `pair_key_multi_split_count = 64`
- `source_speaker_multi_split_count = 9`

This means related pair/speaker patterns appear across splits. Reported performance may therefore be optimistic.

### 7.2 Statistical Coverage

- Baseline has 3 seeds.
- Most extension experiments currently have single-seed results.

This limits confidence for extension-level conclusions.

## 8. Final Conclusion

This project successfully built a working SpecRNet-style pipeline from audio preprocessing to test evaluation and report-ready visualization. The paper-aligned baseline is strong and stable. Extension studies show that attention removal has limited impact on this dataset, GAP improves deployment efficiency with modest quality loss, and short utterances remain significantly harder to detect. Weighted loss did not provide a clear gain in the current setup.

For stronger scientific validity, the next priority is a group-aware split strategy to reduce leakage risk, followed by multi-seed extension experiments.

## 9. Artifact Location

All final tables and figures are under:

`reports/int6068_final_package/`

## 10. Code Usage Guide

This section provides a practical command sequence to reproduce the pipeline.

### 10.1 Environment Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

### 10.2 Feature Extraction

Generate 4-second LFCC features:

```powershell
python extract_lfcc.py --metadata_csv metadata_kaggle.csv --output_dir lfcc_4s --duration 4.0 --sr 16000 --n_lfcc 80 --n_frames 404
```

Generate 1-second LFCC features (short-utterance experiment):

```powershell
python extract_lfcc.py --metadata_csv metadata_kaggle.csv --output_dir lfcc_1s --duration 1.0 --sr 16000 --n_lfcc 80 --n_frames 404
```

### 10.3 Leakage Audit (Recommended Before Training)

```powershell
python scripts/audit_metadata_leakage.py --metadata metadata_kaggle.csv --out_dir reports/leakage_audit_tonight
```

### 10.4 Smoke Test

```powershell
python train_specrnet.py --variant default --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_smoke --batch_size 32 --epochs 1 --lr 1e-4 --weight_decay 1e-4 --scheduler none
```

### 10.5 Paper-Aligned Baseline Training (3 Seeds)

```powershell
python train_specrnet.py --variant default --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_paper_aligned --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant default --seed 123 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_paper_aligned --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant default --seed 3407 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_paper_aligned --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
```

### 10.6 Extension Training

```powershell
python train_specrnet.py --variant no-att --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_extensions --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant gap --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_extensions --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant default --seed 42 --metadata metadata_kaggle.csv --features lfcc_1s --out_dir results_duration --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant default --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_weighted_control --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none
python train_specrnet.py --variant default --seed 42 --metadata metadata_kaggle.csv --features lfcc_4s --out_dir results_weighted --batch_size 128 --epochs 10 --lr 1e-4 --weight_decay 1e-4 --scheduler none --weighted_loss
```

### 10.7 Test Evaluation

Checkpoint naming rule:

`{out_dir}/{variant}/seed_{seed}/best_specrnet_{variant}_seed{seed}.pt`

Examples:

```powershell
python evaluate.py --checkpoint results_paper_aligned/default/seed_42/best_specrnet_default_seed42.pt --metadata metadata_kaggle.csv --features lfcc_4s --split test --output_dir results_eval/default_s42 --variant default --seed 42
python evaluate.py --checkpoint results_extensions/no-att/seed_42/best_specrnet_no-att_seed42.pt --metadata metadata_kaggle.csv --features lfcc_4s --split test --output_dir results_eval/noatt_s42 --variant no-att --seed 42
python evaluate.py --checkpoint results_extensions/gap/seed_42/best_specrnet_gap_seed42.pt --metadata metadata_kaggle.csv --features lfcc_4s --split test --output_dir results_eval/gap_s42 --variant gap --seed 42
```

### 10.8 Build Final Report Assets

Generate final package (tables + core figures):

```powershell
python scripts/build_int6068_final_package.py
```

Generate extra visuals (training curves, LFCC visual, params-time scatter):

```powershell
python scripts/build_int6068_extra_visuals.py
```
