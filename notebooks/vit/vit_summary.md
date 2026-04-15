# ViT Experiments

Binary melanoma classification (HAM10000) using pretrained ViT-B/16. This series explores how loss choice and staged fine-tuning affect performance on an imbalanced dermoscopy dataset.

---

## Benchmark to Beat

Best result from ViT experiments:

| Model | Config | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|
| Focal + Staged ViT (02) | Focal Loss + Head Warm-up + Full Fine-Tuning | **0.6314** | 0.7953 | 0.3461 | 0.9042 |

**Target: Improve AUC beyond 0.9066 while keeping recall at a screening-friendly level**

---

## Experimental Setup

All ViT experiments use:

- **Backbone**: pretrained `ViT-B/16`
- **Input size**: `224 x 224`
- **Batch size**: `32`
- **Augmentation**: shared augmented training transforms
- **Optimiser**: Adam
- **Evaluation**: checkpoint selected by best validation AUC, then threshold tuned on validation F2

---

## Results Summary

| # | Notebook | Key Change | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|
| 01 | `01.vit_b16_weighted`  | Weighted BCE baseline | 0.6005 | **0.8421** | 0.2796 | 0.8837 |
| 02 | `02.vit_b16_focal_staged_finetune`  | + focal loss + staged fine-tuning | **0.6314** | 0.7953 | 0.3461 | 0.9042 |
| 03 | `03.vit_b16_weighted_staged_finetune`. | keep weighted BCE, + staged fine-tuning | 0.6256 | 0.7251 | **0.4039** | **0.9066** |

---

## Key Findings

### 1. Weighted ViT is already a strong screening baseline (01)

- The baseline achieves strong ranking performance:
  - AUC reaches **0.8837**
- It is highly recall-focused:
  - Melanoma recall reaches **0.8421**
- But precision is weak:
  - Melanoma precision is only 0.2796
- So the baseline is useful for sensitivity, but not yet well-balanced

---

### 2. Focal loss + staged fine-tuning gives the best overall tradeoff (01 -> 02)

- AUC improves substantially:
  - **0.8837 -> 0.9042**
- F2 improves clearly:
  - **0.6005 -> 0.6314**
- Precision improves meaningfully:
  - **0.2796 -> 0.3461**

However:
- Recall drops from 0.8421 to 0.7953

- This is the best overall screening-oriented ViT so far because it improves balance without sacrificing too much sensitivity

---

### 3. Staged fine-tuning helps even without focal loss (01 -> 03)

- AUC improves further:
  - **0.8837 -> 0.9066**
- F2 also improves:
  - **0.6005 -> 0.6256**
- Precision improves the most:
  - **0.2796 -> 0.4039**

However:
- Recall drops sharply to 0.7251

- This confirms the staged optimization schedule itself is helpful, but weighted BCE under this setup becomes too conservative for screening

---

### 4. Focal loss gives a better screening balance than weighted BCE under the same staged setup (02 -> 03)

- `03` has slightly better AUC:
  - **0.9042 -> 0.9066**
- `03` also has better precision:
  - **0.3461 -> 0.4039**

But:
- Recall drops from 0.7953 to 0.7251
- F2 drops from 0.6314 to 0.6256

- So focal loss is not the main reason AUC improved, but it does help keep the precision-recall balance better aligned with a screening objective

---

### 5. AUC vs F2 tradeoff

- **Model 03 (Weighted + Staged ViT)**:
  - Best AUC -> strongest raw ranking and best precision
- **Model 02 (Focal + Staged ViT)**:
  - Best F2 -> best threshold-specific screening tradeoff
- **Model 01 (Weighted ViT baseline)**:
  - Best recall -> most sensitivity-focused operating point

---

## Progression

AUC-ROC: 0.8837 -> 0.9042 -> 0.9066  
F2 Score: 0.6005 -> 0.6314 -> 0.6256  

---

## Overall Conclusion

- Best model (AUC):
  -> **03 (Weighted + Staged ViT)**

- Best model (F2):
  -> **02 (Focal + Staged ViT)**

- Best model (Recall):
  -> **01 (Weighted ViT baseline)**

---
