# ResNet-18 Experiments

Binary melanoma classification (HAM10000) using pretrained ResNet-18, progressively exploring backbone unfreezing, data augmentation, and modern regularization techniques.

---

## Benchmark to Beat

Best result from `notebooks/penalty_experiments/`:

| Model | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC | Notebook |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | Aug + L1 (30ep, CosineAnnealingLR) | 0.6689 | **0.6879** | 0.8070 | 0.4326 | — | `efficientnet_b0_aug_l1_l2` |

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
3. **Robustness** — AUC provides a more stable signal of learning progress compared to F2, which can be volatile on small validation sets.

### Why tune threshold by F2 (not recall with a precision floor)

After saving the best-AUC model, the threshold is tuned to maximise **F2 on the validation set**.

- **Maximise F2**: weights recall 2x over precision, matching the clinical priority of melanoma screening (minimising false negatives) while still penalising degenerate all-positive predictions.

### Pipeline

1. **Model selection**: save checkpoint with best val AUC (strongest feature extractor).
2. **Threshold tuning**: sweep thresholds on val set, pick the one that maximises F2.
3. **Reporting**: report all metrics (AUC, Recall, Precision, F1, F2, Balanced Accuracy) on the test set for comparison.

---

## Results Summary

| # | Notebook | Unfrozen | Trainable Params | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.resnet_naive` | None (FC only) | 513 | None | 0.7068 | 0.6409 | 0.7076 | 0.4654 | 0.8900 |
| 02 | `02.resnet_unfreeze` | Full backbone | 11.2M | None | 0.6893 | 0.6280 | 0.7544 | 0.3761 | 0.8997 |
| 03 | `03.resnet_augmented_v1` | Full backbone | 11.2M | Strong Aug + WD=1e-4 + Scheduler | 0.6680 | 0.6660 | **0.8304** | 0.3717 | 0.9068 |
| 04 | `04.resnet_augmented_v2` | Full backbone | 11.2M | Dropout=0.4 + AdamW + TTA | 0.6835 | **0.6704** | 0.7778 | **0.4318** | **0.9133** |

---

## Key Findings

- **Naive Transferability (01)**: An AUC-ROC of 0.8900 with only 513 trainable parameters confirms that ResNet-18's ImageNet features are remarkably descriptive for dermatological images, though precision is limited.
- **The Overfitting Challenge (02)**: Unfreezing the backbone without regularization increased recall but led to a drop in Test F2 and Precision, highlighting the model's tendency to memorize training samples in the absence of constraints.
- **Impact of Augmentation (03)**: Introducing strong augmentation and a cosine scheduler significantly boosted melanoma recall to **0.8304**—the highest in this series—confirming that orientation-invariant transforms are crucial for skin lesion classification.
- **Inference Stability (04)**: The combination of Dropout, AdamW, and Test-Time Augmentation (TTA) yielded the best discriminative performance (**AUC 0.9133**) and the highest overall Test F2 (**0.6704**). TTA successfully stabilized predictions, recovering precision lost in earlier iterations.

---

## Progression

```
Test F2:  0.641 → 0.628 → 0.666 → 0.670
AUC-ROC: 0.890 → 0.900 → 0.907 → 0.913
              Iteration 04 (Dropout + AdamW + TTA) is the strongest ResNet-18 configuration.
              Iteration 03 provides the highest sensitivity (Recall 0.83).
```
