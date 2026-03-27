
````markdown
# SpecRNet Reproduction for Synthetic Audio Detection

This repository contains a **reproduction-oriented implementation** of **SpecRNet**, a lightweight neural network for detecting synthetic speech (audio deepfakes).  
The project was developed for the **INT6068 Neural Networks and Deep Learning** coursework.

> This project follows the **core SpecRNet architecture** and its lightweight benchmarking philosophy, while additionally implementing a custom **data preparation**, **LFCC extraction**, **training**, and **evaluation** pipeline on public datasets.

---

## 🌟 Highlights

- **Lightweight model**: only **278,092 trainable parameters**
- **Fast inference**: approximately **2.00 ms/sample**
- **LFCC-based frontend**: fixed-size **80 × 404** LFCC representation
- **Multiple evaluation settings**:
  - same-domain testing
  - cross-generator testing
  - in-the-wild / Kaggle testing
  - Gaussian-noise robustness testing

---

## 📊 Main Experimental Findings

### 1. Split Integrity
- Training, validation, and testing sets were verified to have **zero overlap**, indicating **no data leakage**.

### 2. Same-Domain Performance
- On the LJSpeech-based controlled benchmark (real speech vs. HiFi-GAN synthetic speech), the model achieved **near-saturated performance**.

### 3. Cross-Generator Generalization
- When evaluated on **unseen vocoders**, performance dropped compared with the same-domain setting, revealing a clear **cross-generator generalization gap**.
- Results remained very strong on several LJSpeech-based generators, while performance was lower on out-of-domain generators such as **JSUT-based** systems.

### 4. In-the-Wild / Kaggle Evaluation
- On external open-domain data, performance dropped substantially compared with laboratory-style benchmarks.
- This indicates a clear **domain shift challenge** caused by differences in recording conditions, language, compression, and unseen synthesis patterns.

### 5. Noise Robustness (Qualified)
- Under **moderate Gaussian noise**, the model remained highly accurate:
  - **Clean**: ≈ **0.9997**
  - **40 dB**: ≈ **0.9995**
  - **30 dB**: ≈ **0.9990**
  - **20 dB**: ≈ **0.9852**
- Under **strong noise**, performance degraded sharply:
  - **10 dB and below**: approximately **0.50** (near random guessing)
- Therefore, the model is robust to **moderate** noise, but not to **severe** noise.

### 6. Long-Form Audio Inference
- The sliding-window LFCC pipeline was successfully executed on a **large long-form benchmark** without obvious runtime instability under the tested setup.

---

## 📁 Project Structure

- `build_metadata.py` — build paired metadata from real/fake audio
- `extract_lfcc.py` — LFCC feature extraction (80 × 404)
- `train_specrnet.py` — training pipeline with validation and checkpoint saving
- `evaluate.py` — full-dataset evaluation (Accuracy, Precision, Recall, F1, AUC, confusion matrix, ROC)
- `dataset.py` — PyTorch dataset loader
- `config.py` — model configuration
- `best_specrnet.pt` — trained model weights
- `results/` — saved evaluation outputs and plots

---

## 🚀 Getting Started

### 1. Environment
Install required packages:

```bash
pip install torch librosa numpy pandas scipy tqdm scikit-learn matplotlib seaborn
````

### 2. Build metadata

Example for same-domain LJSpeech vs HiFi-GAN setup:

```bash
python build_metadata.py --real_dir LJSpeech-1.1/wavs --fake_dir generated_audio/ljspeech_hifiGAN --output_csv metadata_final.csv
```

### 3. Extract LFCC features

```bash
python extract_lfcc.py --metadata_csv metadata_final.csv --output_dir ./lfcc_features
```

### 4. Train

```bash
python train_specrnet.py
```

### 5. Evaluate (same-domain)

```bash
python evaluate.py --checkpoint best_specrnet.pt --metadata metadata_final.csv --features lfcc_features --split test --output_dir results_final
```

### 6. Evaluate (cross-generator / external)

```bash
python evaluate.py --checkpoint best_specrnet.pt --metadata metadata_cross_test.csv --features lfcc_cross --split test --output_dir results_cross
```

***

## 🧠 Discussion

This project reproduces the **core SpecRNet architecture** and validates the original paper’s lightweight design philosophy: a compact neural network with fast inference and strong detection ability.

The experiments reveal three important patterns:

1.  **Near-perfect in-domain performance**  
    When trained and tested within the same controlled domain (LJSpeech-based synthetic speech), the detector achieves extremely high performance.

2.  **Generalization gap across generators and domains**  
    Performance declines when the model is evaluated on unseen generators or external datasets, indicating that the detector partially relies on **generator-specific** and **domain-specific** artifacts.

3.  **Noise robustness has a clear boundary**  
    The detector remains highly reliable under **moderate Gaussian noise**, but degrades sharply under **strong noise**. Therefore, robustness should be interpreted as **SNR-dependent**, rather than universal.

These findings suggest that SpecRNet is highly effective in controlled conditions, but additional work is required for robust deployment in unconstrained real-world environments.

***

## ⚠️ Limitations

*   This project is **not a strict 1:1 reproduction** of the original paper’s full experimental protocol.
*   The official SpecRNet implementation mainly provides the **architecture** and **benchmark scripts**, while this project additionally implements the custom preprocessing, LFCC extraction, training, and evaluation workflow.
*   Results on noisy and external datasets should be interpreted within the tested setup:
    *   noise experiments currently use **Gaussian noise**
    *   long-form inference stability was verified only under the current sliding-window pipeline and datasets
    *   cross-domain results depend on the chosen external data distributions

***

## ✍️ Contribution

**DENG JIAJUN**

*   model training and reproduction pipeline
*   LFCC frontend implementation
*   same-domain, cross-generator, and in-the-wild benchmarking
*   robustness analysis under Gaussian noise

````

---

