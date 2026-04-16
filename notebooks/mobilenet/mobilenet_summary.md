# MobileNet Experiments

Binary melanoma classification (HAM10000) using pretrained MobileNetV3-Large, exploring metadata fusion and TTA. All notebooks evaluated on the standard test split (1511 samples: 1340 nevus, 171 melanoma).

---

## Benchmark to Beat

Best result from prior experiments:

| Model | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC | Notebook |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | L1 + Dropout + Metadata + TTA (8×) | 0.6887 | **0.6952** | 0.8830 | 0.3756 | 0.9184 | `efficientnet/06` |

**Target: Test F2 > 0.6952**

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
3. **Robustness** — AUC provides a more stable signal of learning progress compared to F2, which can be volatile on small validation sets.

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
- **Augmentation**: Strong (HFlip + VFlip + Rotation(30°) + ColorJitter + RandomAffine) for training; eval transforms for val/test.
- **Optimiser**: AdamW.
- **Scheduler**: `CosineAnnealingLR`.
- **Evaluation**: Threshold tuning on val set (max F2), reported on test set with 8× TTA.

---

## Results Summary

| # | Notebook | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 01 | `01.mobilenet_v3_metadata_tta` | Full backbone, L1+L2 (λ=1e-3), Dropout=0.5, Metadata, TTA (8×) | 0.6802 | 0.6492 | 0.8421 | 0.3388 | 0.9114 |
| 02 | `02.mobilenet_v3_efficientnet_b0_ensemble` | Ensemble (MobileNetV3-Large Iter 01 + EfficientNet-B0 Iter 06), mean TTA probs (8×) | 0.7023 | 0.6781 | 0.8772 | 0.3555 | 0.9235 |

---

## Key Findings

- **Below benchmark (01)**: MobileNetV3-Large with full unfreezing, dual regularisation (L1+L2=1e-3), metadata fusion, and 8× TTA achieved Test F2 0.6492 — below both the 0.6879 benchmark and EfficientNet-B0's best of 0.6952. AUC-ROC (0.9114) also trails EfficientNet-B0 Iter 05 (0.9199).
- **Persistent overfitting (01)**: Despite L1+L2 regularisation, training AUC continued climbing to 0.9556 by epoch 30 while val AUC plateaued at ~0.915 after the best checkpoint at epoch 16. The early best checkpoint (ep16/30) suggests the model is still memorising rather than generalising — stronger dropout or partial unfreezing may be needed.
- **Val→Test generalisation gap (01)**: Val F2 (TTA) was 0.6802 but Test F2 dropped to 0.6492, a gap of 0.031. This is larger than EfficientNet-B0's gap (~0.001 for Iter 06), pointing to weaker generalisation — likely because MobileNetV3-Large's backbone was trained with a very different inductive bias (efficiency-focused depthwise separable convolutions + SE blocks) that may be less suited to dermoscopic textures than EfficientNet.
- **Smaller backbone than expected (01)**: The ~3M trainable params reflect only the `features` + `avgpool` portion of the model — the original MobileNetV3-Large classifier (Linear(960→1280→1000), ~2.5M params) was discarded. Including the pre-classifier expansion layer before fusion could provide a richer 1280-dim image embedding and may close the gap with EfficientNet-B0.
- **Ensemble beats AUC but not Test F2 (02)**: Mean-averaging TTA probabilities from MobileNetV3-Large and EfficientNet-B0 raised AUC-ROC to 0.9235 (vs. 0.9182 for EfficientNet-B0 alone) and Val F2 to 0.7023, but Test F2 of 0.6781 falls short of EfficientNet-B0's standalone 0.6952. The ensemble generalises better in AUC but the MobileNet component pulls down Test F2 — its weaker individual performance (0.6492) drags the averaged probabilities away from the optimal threshold, reducing precision without a compensating recall gain.

---

## Progression

```
Test F2:  0.649 → 0.678
AUC-ROC: 0.911 → 0.924

  Iter 01: MobileNetV3-Large alone does not beat the benchmark (0.6879) or EfficientNet-B0 TTA (0.6952).
  Iter 02: Ensemble improves AUC (0.9235) and Val F2 (0.7023) but Test F2 (0.6781) still trails
           EfficientNet-B0 standalone (0.6952). Weaker MobileNet component limits Test F2 gains.
```
