# DenseNet-121 Experiments

Binary melanoma classification (HAM10000) using DenseNet-121, progressively exploring pretraining, head architecture, and regularisation strategies.

---

## Benchmark to Beat

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

1. **AUC is threshold-independent** — it measures the model's raw discriminative ability, which is the fundamental bottleneck (our best: 0.858 vs ISIC SOTA: 0.973). F2, recall, precision are all downstream of AUC and depend on the decision threshold.
2. **AUC is the ceiling** — a higher AUC model will always have a better precision-recall tradeoff available at some threshold. Improving AUC lifts all metrics.
3. **ISIC benchmarks confirm this** — the competition winners with AUC 0.97 achieved high scores on every metric simultaneously. At that AUC level, the threshold barely matters.

### Why tune threshold by F2 (not recall with a precision floor)

After saving the best-AUC model, the threshold is tuned to maximise **F2 on the validation set**. Alternatives considered:

- **Maximise recall with a precision floor (e.g., >= 0.20)**: rejected because it tanks F1 and balanced accuracy. With our current AUC (~0.85), there is a fundamental precision-recall tradeoff — pushing recall high forces precision down, giving poor F1. The ISIC benchmarks achieve F1 = 93% because their AUC is high enough to sustain both.
- **Maximise F1**: gives equal weight to precision and recall. Reasonable, but for melanoma detection recall should be weighted higher (false negatives are more costly than false positives).
- **Maximise F2**: weights recall 2x over precision, matching the clinical priority of melanoma screening while still penalising degenerate all-positive predictions.

### Pipeline

1. **Model selection**: save checkpoint with best val AUC (strongest feature extractor)
2. **Threshold tuning**: sweep thresholds on val set, pick the one that maximises F2
3. **Reporting**: report all metrics (AUC, Recall, Precision, F1, F2, Balanced Accuracy) for honest comparison against ISIC benchmarks

---

All experiments use:
- **Loss**: BCEWithLogitsLoss (with pos_weight ~8.1 where noted)
- **Augmentation**: HFlip + VFlip + Rotation(30°) + ColorJitter + RandomAffine
- **Optimiser**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 30
- **Evaluation**: threshold tuning on val set, reported on test set

---

## Results Summary

| # | Notebook | Pretrained | Frozen | Head Structure | Hidden Dim | Dropout | Pos Weight | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.densenet121_baseline` | False | False | Linear | 0 | 0.0 | None | 0.5997 | 0.5771 | 0.8012 | 0.2724 | 0.8570 |
| 02 | `02.densenet121_baseline_weighted` | False | False | Linear | 0 | 0.0 | ✓ | 0.5945 | 0.5837 | 0.8889 | 0.2460 | 0.8531 |
| 03 | `03.densenet121_batchnorm_weighted` | False | False | Linear → BN → ReLU → Linear | 256 | 0.0 | ✓ | 0.5978 | **0.5964** | 0.8246 | 0.2831 | **0.8580** |
| 04 | `04.densenet121_dropout_weighted` | False | False | Linear → ReLU → Dropout → Linear | 256 | 0.5 | ✓ | 0.6122 | 0.5882 | 0.8889 | 0.2500 | 0.8543 |
| 05 | `05.densenet121_pretrained_weighted` | True | True | Linear → ReLU → Dropout → Linear | 256 | 0.4 | ✓ | 0.5814 | 0.5516 | 0.8070 | 0.2434 | 0.8449 |

## Detailed Results

| # | Threshold | AUC-ROC | Bal. Accuracy | Test F2 | Test Recall | Test Precision | Test F1 |
|---|---|---|---|---|---|---|---|
| 01 | 0.17 | 0.8570 | 0.7640 | 0.5771 | 0.8012 | 0.2724 | 0.4065 |
| 02 | 0.54 | 0.8531 | 0.7706 | 0.5837 | 0.8889 | 0.2460 | 0.3853 |
| 03 | 0.63 | 0.8580 | 0.7791 | **0.5964** | 0.8246 | 0.2831 | 0.4215 |
| 04 | 0.56 | 0.8543 | 0.7743 | 0.5882 | 0.8889 | 0.2500 | 0.3902 |
| 05 | 0.68 | 0.8449 | 0.7434 | 0.5516 | 0.8070 | 0.2434 | 0.3740 |

---

## Key Findings

- **Unweighted baseline (01)**: AUC-ROC 0.857 confirms DenseNet-121 ImageNet features show reasonable transferability to dermoscopy even from scratch, but with no pos_weight the model under-prioritises the minority class — low threshold (0.17) needed to recover recall.
- **Adding pos_weight (02)**: Loss reweighting alone meaningfully improves recall (0.8012 → 0.8889) and balanced accuracy (0.764 → 0.771), confirming class imbalance was a key bottleneck. AUC dips marginally (0.857 → 0.853), likely due to the loss landscape shift rather than reduced discriminative ability.
- **BatchNorm head (03)**: Best overall result — highest test F2 (0.5964), highest AUC (0.858), and highest balanced accuracy (0.779). The hidden layer with batch normalisation adds capacity and stabilises activations without overfitting, outperforming all other configurations on test metrics.
- **Dropout head (04)**: Dropout at 0.5 matches pos_weight weighted recall (0.8889) but achieves slightly lower F2 (0.5882) and AUC (0.854) compared to BatchNorm (03). Stochastic regularisation is less effective than normalisation for this dataset size when training from scratch.
- **Pretrained frozen backbone (05)**: Counterintuitively, the worst-performing configuration (Test F2 0.5516, AUC 0.8449). Freezing the ImageNet backbone limits adaptation to dermoscopy-specific features — the frozen features from a natural image domain do not transfer as strongly to dermoscopic images as expected, and the small trainable head lacks capacity to compensate.

---

## Progression

```
Test F2:  0.5771 → 0.5837 → 0.5964 → 0.5882 → 0.5516
AUC-ROC: 0.8570 → 0.8531 → 0.8580 → 0.8543 → 0.8449
              BatchNorm head + pos_weight (03) is best on all primary metrics
              Pretrained frozen (05) underperforms all from-scratch variants
```

---

