# CNN Experiments

Binary melanoma classification (HAM10000) using custom CNN architectures. This series explores how model capacity, normalization, and residual connections affect performance on an imbalanced medical imaging dataset.

---

## Benchmark to Beat

Best result from CNN experiments:

| Model | Config | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|
| Residual CNN (05) | BatchNorm + Residual + Weighted | 0.5814 | 0.8187 | 0.2692 | **0.8672** |

**Target: Improve AUC beyond 0.8672**

---

## Experimental Setup

All CNN experiments use:

- **Loss**: BCEWithLogitsLoss with `pos_weight`
- **Class imbalance handling**: weighted loss
- **Optimiser**: Adam 
- **Augmentation**: consistent across experiments
- **Evaluation**: threshold tuning on validation set

---

## Results Summary

| # | Notebook | Architecture | Key Change | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 01 | `01.cnn_baseline` | Simple CNN | No weighting, no BN | 0.5676 | 0.8246 | 0.2527 | 0.8449 |
| 02 | `02.cnn_baseline_weighted` | Simple CNN | + pos_weight | 0.5670 | **0.8713** | 0.2365 | 0.8471 |
| 03 | `03.cnn_batchnorm_weighted` | CNN + BatchNorm | + BatchNorm | **0.5900** | 0.7895 | 0.2935 | 0.8520 |
| 04 | `04.cnn_deeper_batchnorm_weighted` | Deeper CNN + BN | + depth | 0.5525 | 0.7076 | 0.2944 | 0.8538 |
| 05 | `05.cnn_residual_batchnorm_weighted` | Residual CNN | + skip connections | 0.5814 | **0.8187** | 0.2692 | **0.8672** |

---

## Key Findings

### 1. Class imbalance handling mainly shifts the operating point (01 → 02)

- Introducing `pos_weight` makes the baseline much more recall-focused:
  - Recall improves from 0.8246 to **0.8713**
- But the trade-off is weaker precision and almost no F2 gain:
  - Precision drops from 0.2527 to 0.2365
  - F2 is essentially flat (0.5676 → 0.5670)
- Weighting helps screening sensitivity, but not overall balance by itself

---

### 2. BatchNorm improves stability and performance (02 → 03)

- More stable training
- Better generalisation
- Better precision-recall balance than the rerun weighted baseline
- Leads to **best F2 score (0.5900)**

---

### 3. Increasing depth alone does not help (03 → 04)

- Slight AUC increase (0.8520 → 0.8538)
- But:
  - Recall drops significantly
  - F2 decreases

---

### 4. Residual connections improve representation learning (04 → 05)

- Largest AUC improvement:
  - **0.8538 → 0.8672**
- Highest recall:
  - **0.8187**

However:
- Precision decreases
- F2 slightly below best (03)

---

### 5. AUC vs F2 tradeoff

- **Model 05 (Residual CNN)**:
  - Best AUC → best feature extractor
- **Model 03 (BatchNorm CNN)**:
  - Best F2 → best threshold-specific tradeoff

---

## Progression

AUC-ROC: 0.8449 → 0.8471 → 0.8520 → 0.8538 → 0.8672  
F2 Score: 0.5676 → 0.5670 → 0.5900 → 0.5525 → 0.5814  

---

## Overall Conclusion

- Best model (AUC):
  → **05 (Residual CNN)**

- Best model (F2):
  → **03 (BatchNorm CNN)**

---

## Next Direction

- Transition to transfer learning (ResNet / EfficientNet)
- Improve AUC beyond 0.88–0.90
