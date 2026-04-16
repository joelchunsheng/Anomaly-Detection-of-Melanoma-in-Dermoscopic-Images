# EfficientNet Experiments

Binary melanoma classification (HAM10000) using pretrained EfficientNet variants, progressively exploring backbone unfreezing, regularisation, and patient metadata fusion. All notebooks have been evaluated on the standard test split (1511 samples: 1340 nevus, 171 melanoma).

---

## Benchmark to Beat

Best result from `notebooks/penalty_experiments/`:

| Model | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC | Notebook |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | Aug + L1 (n=6, 30ep, CosineAnnealingLR) | 0.6689 | **0.6879** | 0.8070 | 0.4326 | — | `efficientnet_b0_aug_l1_l2` |

**Target: Test F2 > 0.6879**

### ISIC 2018 Competition & State-of-the-Art Reference

| Source | Model | Melanoma Recall | AUC | Balanced Acc | F1 |
|---|---|---|---|---|---|
| ISIC 2018 Winner | — | 88.5% | 0.971 | — | — |
| SOTA | Xception + Deep Attention | — | — | 95.9% | — |
| SOTA | ResNet50 + EfficientNetB0 + Patient Metadata Fusion | 93% | >0.973 | — | 93% |

---

## Model Selection & Evaluation Strategy

### Why save by best val AUC (not F2)

Models are saved based on **best validation AUC-ROC** instead of best val F2. Rationale:

1. **AUC is threshold-independent** — it measures the model's raw discriminative ability, which is the fundamental bottleneck. F2, recall, and precision are all downstream of AUC and depend on the decision threshold.
2. **AUC is the ceiling** — a higher AUC model will always have a better precision-recall tradeoff available at some threshold. Improving AUC lifts all metrics.
3. **Robustness** — F2 can be highly volatile during training on small validation sets; AUC provides a more stable signal of learning progress.

### Why tune threshold by F2 (not recall with a precision floor)

After saving the best-AUC model, the threshold is tuned to maximise **F2 on the validation set**. 

- **Maximise F2**: weights recall 2x over precision, matching the clinical priority of melanoma screening (minimising false negatives) while still penalising degenerate all-positive predictions.

### Pipeline

1. **Model selection**: save checkpoint with best val AUC (strongest feature extractor).
2. **Threshold tuning**: sweep thresholds on val set, pick the one that maximises F2.
3. **Reporting**: report all metrics (AUC, Recall, Precision, F1, F2, Balanced Accuracy) on the test set for comparison.

---

All experiments use:
- **Loss**: `BCEWithLogitsLoss` with `pos_weight` (~8.1) for class imbalance.
- **Augmentation**: Varies (Basic: HFlip + Rotation(10°); Strong: HFlip + VFlip + Rotation(30°) + ColorJitter + RandomAffine).
- **Optimiser**: Adam or AdamW.
- **Scheduler**: `CosineAnnealingLR` (where indicated).
- **Evaluation**: Threshold tuning on val set (max F2), reported on test set.

---

## Results Summary

| # | Notebook | Variant | Unfrozen | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.efficientnet_b3` | B3 | Full Backbone | Basic Aug | 0.6685 | 0.6490 | 0.8304 | 0.3463 | 0.9029 |
| 02 | `02.efficientnet_b3_augment` | B3 | Full Backbone | Strong Aug + AdamW | 0.6719 | 0.6683 | 0.7778 | 0.4277 | 0.9070 |
| 03 | `03.efficientnet_b0` | B0 | Full Backbone | Basic Aug | 0.6716 | 0.6455 | 0.7836 | 0.3785 | 0.9077 |
| 04 | `04.efficientnet_b3_partial_unfreeze` | B3 | Last 3 Blocks | Basic Aug | 0.6913 | 0.6220 | 0.7661 | 0.3550 | 0.8743 |
| 05 | `05.efficientnet_b0_l1_metadata` | B0 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata | 0.6872 | 0.6875 | 0.8363 | 0.4017 | **0.9199** |
| 06 | `06.efficientnet_b0_l1_metadata_tta` | B0 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata + TTA (8×) — train + TTA in one notebook | 0.6873 | **0.6952 ✓** | **0.8830** | 0.3756 | 0.9182 |
| 07 | `07.efficientnet_b0_label_smooth` | B0 | Last 6 Blocks | Label Smooth (ε=0.1) + 1:7 WRS + pos_weight=7 + Dropout 0.5 + L1 + Metadata | 0.6526 | 0.6138 | 0.7602 | 0.3467 | 0.8869 |
| 08 | `08.efficientnet_b0_focal_loss` | B0 | Last 6 Blocks | Focal Loss (γ=2.0, α=0.89) + Dropout 0.5 + L1 + Metadata | 0.6092 | 0.5734 | 0.7953 | 0.2709 | 0.8501 |
| 09 | `09.efficientnet_b4_metadata` | B4 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata @ 380×380 | 0.6751 | 0.6611 | 0.7895 | 0.4006 | 0.9014 |
| 10 | `10.efficientnet_b4_metadata_tta` | B4 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata + TTA (8×) @ 380×380 | 0.6731 | 0.6782 | 0.7544 | **0.4831** | 0.9058 |

---

## Key Findings

- **Architecture Efficiency**: EfficientNet variants consistently achieve higher AUC-ROC (>0.90) than ResNet-50 (~0.88), confirming the superiority of the architecture for dermatological feature extraction.
- **Metadata Fusion (05)**: Incorporating patient metadata (age, sex, body site) pushed the AUC-ROC to its highest recorded value (**0.9199**). While F2 remains tied with the non-metadata baseline, the model is better calibrated and achieves superior recall (0.8363).
- **Overfitting and Unfreezing (04)**: Restricting unfreezing to just 3 blocks in B3 led to a high validation F2 but poor test performance, suggesting that deeper fine-tuning is necessary for EfficientNet to adapt to dermoscopic images.
- **Regularisation Impact**: Strong augmentation and AdamW (02) significantly improved precision (0.4277 vs 0.3463) compared to basic augmentation, though at a slight cost to recall.
- **TTA (06)**: Applying 8-augment TTA to the Iter 05 checkpoint — with no additional training — pushed Test F2 from 0.6875 to **0.6952**, beating the 0.6879 benchmark. TTA averaging lowered the optimal threshold from 0.62 to 0.54 and boosted recall from 0.8363 to **0.8830** at a small precision cost. AUC is marginally lower (0.9184 vs 0.9199) as expected — TTA affects calibration but not the underlying ranking ability.
- **Label Smoothing + Sampler (07)**: Label smoothing (ε=0.1) combined with a 1:7 WeightedRandomSampler and pos_weight=7.0 significantly hurt generalisation. AUC dropped to 0.8869 and Test F2 to 0.6138 — worse than the raw Iter 05 baseline. The sampler enforced a tighter class ratio in each batch, reducing the effective signal from the majority class, while the combination of label smoothing and reduced pos_weight under-penalised false negatives. The val→test F2 gap (0.6526 → 0.6138) is larger than any prior iteration.
- **Focal Loss (08)**: Focal Loss with γ=2.0 was too aggressive for this dataset. By heavily down-weighting easy (well-classified) samples, the loss provided insufficient gradient signal for convergence — AUC reached only **0.8501** (the lowest in the series) and Test F2 collapsed to **0.5734**. The extreme class imbalance (1:8.1) means most easy samples are nevus; suppressing them starves the model of the background signal it needs to distinguish hard cases. γ=1.0 or lower would be worth exploring.
- **EfficientNet-B4 @ 380×380 (09)**: Scaling up to B4 at native resolution (380×380) underperformed B0 Iter 05 across all metrics — Test F2 0.6611 vs 0.6875, AUC 0.9014 vs 0.9199. The root cause is over-regularisation: L1=1e-3 is calibrated for B0's 4M params but constrains B4's 17.5M params too aggressively. Train AUC (~0.89) stayed consistently *below* val AUC (peak 0.9100) throughout training, a clear sign of underfitting rather than the expected overfitting. The noisy val AUC curve and early best checkpoint (ep19/30) suggest the learning rate schedule needs adjustment for this larger model. The high optimal threshold (0.70 vs B0's 0.62) indicates the model is under-confident on positives. A weaker L1 penalty (e.g. 1e-4) or lower backbone LR with warmup would likely improve results.
- **EfficientNet-B4 TTA @ 380×380 (10)**: Applying 8× TTA to the Iter 09 B4 checkpoint improved Test F2 from 0.6611 to **0.6782** (+0.017). However, TTA behaved *oppositely* to B0: the optimal threshold rose from 0.70 to 0.75 (vs B0's drop from 0.62 to 0.54), recall fell (0.7895 → 0.7544), and precision rose sharply (0.4006 → **0.4831** — the highest precision in the series). This suggests ColorJitter at 380×380 resolution introduces enough variance that TTA averaging pushes borderline positive probabilities *lower*, making the classifier more conservative. AUC improved marginally (0.9014 → 0.9058), consistent with TTA reducing noise in the ranking. Despite the F2 gain, B4+TTA (0.6782) still trails B0+TTA (0.6952).

---

## Progression

```
Test F2:  0.649 → 0.668 → 0.646 → 0.622 → 0.688 → 0.695 ✓ → 0.614 → 0.573 → 0.661 → 0.678
AUC-ROC: 0.903 → 0.907 → 0.908 → 0.874 → 0.920 → 0.918 → 0.887 → 0.850 → 0.901 → 0.906

  Iteration 06 (B0 + TTA) remains the strongest: Test F2 0.6952 > benchmark 0.6879
  Iteration 05 holds the highest AUC-ROC (0.9199)
  Iterations 07 and 08 both regressed — label smoothing+sampler and focal loss hurt.
  Iteration 09 (B4 @ 380×380) underperformed B0 — L1=1e-3 over-regularises the 17.5M param model.
  Iteration 10 (B4 + TTA) improved over Iter 09 but still trails B0+TTA — TTA raised threshold and
  precision rather than boosting recall as it did for B0.
  Conclusion: B0+TTA (Iter 06) remains the best single-model strategy found so far.
```
