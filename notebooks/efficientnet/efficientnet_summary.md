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
| 05 | `05.efficientnet_b0_l1_metadata` | B0 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata | 0.6872 | **0.6875** | 0.8363 | 0.4017 | **0.9199** |

---

## Key Findings

- **Architecture Efficiency**: EfficientNet variants consistently achieve higher AUC-ROC (>0.90) than ResNet-50 (~0.88), confirming the superiority of the architecture for dermatological feature extraction.
- **Metadata Fusion (05)**: Incorporating patient metadata (age, sex, body site) pushed the AUC-ROC to its highest recorded value (**0.9199**). While F2 remains tied with the non-metadata baseline, the model is better calibrated and achieves superior recall (0.8363).
- **Overfitting and Unfreezing (04)**: Restricting unfreezing to just 3 blocks in B3 led to a high validation F2 but poor test performance, suggesting that deeper fine-tuning is necessary for EfficientNet to adapt to dermoscopic images.
- **Regularisation Impact**: Strong augmentation and AdamW (02) significantly improved precision (0.4277 vs 0.3463) compared to basic augmentation, though at a slight cost to recall.

---

## Progression

```
Test F2:  0.649 → 0.668 → 0.646 → 0.622 → 0.688
AUC-ROC: 0.903 → 0.907 → 0.908 → 0.874 → 0.920
          (Iteration 05 matches the best known F2 while setting a new AUC-ROC record)
```
