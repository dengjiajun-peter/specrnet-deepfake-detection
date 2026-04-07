# INT6068 Final Interpretation Draft

## 1. Baseline Result Summary (Paper-Aligned Track)

Strict baseline setup:
- variant: default
- seeds: 42, 123, 3407
- epochs: 10
- batch size: 128
- lr: 1e-4
- weight decay: 1e-4
- scheduler: none

Test results (3-seed):
- Accuracy: 0.9863 +- 0.0029
- F1: 0.9922 +- 0.0016
- AUC: 0.9984 +- 0.0011
- EER: 0.0187 +- 0.0042

Interpretation:
- Baseline performance is strong and stable across the 3 seeds.
- This setup is close to paper-style training settings (Adam-like setting with 3 seeds and EER/AUC emphasis).

## 2. Extension Result Summary (Project Track)

Core extension results (test):
- no-att: AUC 0.9991, EER 0.0195, F1 0.9934, Accuracy 0.9884
- gap: AUC 0.9970, EER 0.0277, F1 0.9892, Accuracy 0.9813
- duration_1s: AUC 0.9244, EER 0.1578, F1 0.9622, Accuracy 0.9322
- weighted_loss_off: AUC 0.9981, EER 0.0215, F1 0.9904, Accuracy 0.9831
- weighted_loss_on: AUC 0.9972, EER 0.0328, F1 0.9898, Accuracy 0.9822

Interpretation:
- no-att is comparable to default, suggesting attention is not the primary contributor in this dataset.
- gap remains competitive but with noticeable EER degradation, indicating temporal modeling still helps.
- 1s input causes a clear performance drop, matching short-utterance difficulty narrative.
- weighted loss does not improve this current setup on test metrics.

## 3. Final Tables (Copy-Ready)

Generated files:
- baseline_table.csv
- baseline_mean_std.csv
- extension_table.csv
- speed_complexity_table.csv

Location:
- reports/int6068_final_package/

## 4. Recommended Figures

Generated files:
- ablation_bar_auc_eer.png
- duration_vs_performance.png
- roc_overview.png
- confusion_overview.png

Use in report:
- ROC overview: show discriminative ability of baseline and core ablations.
- Confusion overview: compare weighted loss behavior directly.
- Ablation bars: summarize no-att/gap effect on AUC and EER.
- Duration plot: highlight short-utterance degradation.

## 5. Result Interpretation

Paper-aligned baseline vs project extensions should be reported separately.

Suggested message:
- Reproduction claim: method-level and training-style alignment is reasonable.
- Extension claim: attention removal had limited impact, GAP offers speed gains with modest accuracy cost, and short utterances remain difficult.
- Weighted loss in this run did not provide a clear gain.

## 6. Limitations and Risks

Important risk from leakage audit:
- pair_key_multi_split_count = 64
- source_speaker_multi_split_count = 9

Implication:
- Related pair/speaker patterns appear across splits and may inflate reported test performance.

Also note:
- Extension experiments are currently mostly single-seed (except baseline), so extension-level uncertainty is under-estimated.

## 7. Final Conclusion Draft

This project achieved a strong baseline under paper-aligned training settings and reproduced key trends of SpecRNet-style detection. In project extensions, removing attention had minimal impact in this dataset, while replacing temporal modeling with GAP significantly reduced model size and inference time at some cost in EER. The short-utterance setting (1s) showed substantial degradation, consistent with the expected challenge. Weighted loss did not improve the current run.

Overall, the study is suitable for course-level reporting, but conclusions should explicitly acknowledge leakage risk and recommend group-aware split design plus multi-seed extension runs as immediate future work.
