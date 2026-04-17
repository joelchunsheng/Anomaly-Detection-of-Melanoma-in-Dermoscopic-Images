# Recreating the Best Models from Scratch

This guide describes how to recreate the two model checkpoints used in `notebooks/Group16.ipynb` and reproduce the project's highest AUC score of **0.9235**.

The final result is an ensemble of two independently-trained models:

| Checkpoint | Architecture | Val AUC | Test F2 |
|---|---|---|---|
| `models/efficientnet_b0_l1_metadata_best.pth` | EfficientNet-B0 + metadata | 0.9167 | 0.6952 |
| `models/01.mobilenet_v3_metadata_best.pth` | MobileNetV3-Large + metadata | 0.9188 | 0.6492 |
| **Ensemble (both combined)** | Mean TTA probabilities | **0.9235** | 0.6781 |

---

## Prerequisites

### 1. Environment

Follow the setup instructions in `README.md` to create a virtual environment and install dependencies.

### 2. Dataset

Download and prepare the HAM10000 dataset. Follow the setup instructions in `README.md` (Kaggle CLI required). After running `scripts/setup_data.py`, the following paths must exist:

```
data_new/
├── raw/
│   ├── HAM10000_metadata/
│   └── ISIC2018_Task3_Test_GroundTruth.csv
├── images/
│   ├── train/
│   └── test/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Split sizes: **Train 7,991 | Val 2,024 | Test 1,511**.  
Class imbalance ratio (non-melanoma : melanoma) ≈ **8.1 : 1** — used as `pos_weight` in BCE loss.

---

## Step 1 — Train EfficientNet-B0 with Metadata

**Reference notebook:** `notebooks/efficientnet/06.efficientnet_b0_l1_metadata_tta.ipynb`  
**Output checkpoint:** `models/efficientnet_b0_l1_metadata_best.pth`

### Architecture

The model is defined in PyTorch as `EfficientNetB0WithMetadata` in `src/models/efficientnet.py`. It fuses image and patient features:

- **Image branch:** EfficientNet-B0 pretrained on ImageNet (`features` + `avgpool`) → **1280-dim** feature vector
- **Metadata branch:** `Linear(17, 32) → ReLU` → **32-dim** feature vector
- **Fusion head:** `cat(1280, 32)` → `Dropout(0.5)` → `Linear(1312, 1)`

Metadata dimension 17 encodes: age (1), sex (1), anatomical localization one-hot (15).

The backbone is frozen at instantiation. The **last 6 blocks** of `model.features` are then unfrozen:

```python
model = EfficientNetB0WithMetadata(metadata_dim=17, num_classes=1, freeze_backbone=True, dropout=0.5)
for block in list(model.features)[-6:]:   # features[3..8]: MBConv blocks 3-7 + head conv
    for param in block.parameters():
        param.requires_grad = True
```

Trainable parameters: **~4M / 4M total**.

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR — unfrozen backbone blocks | 1e-4 |
| LR — metadata encoder + head | 1e-3 |
| Weight decay | 1e-3 |
| L1 regularization (λ) | 1e-3 |
| L2 regularization (λ) | 0.0 |
| Dropout | 0.5 |
| Loss | `BCEWithLogitsLoss(pos_weight=8.11)` |
| Scheduler | `CosineAnnealingLR(T_max=30)` |
| Epochs | 30 |
| Batch size | 32 |
| Image size | 224×224 |
| Seed | 42 |

### Training Augmentation

Applied via `get_augmented_train_transforms(image_size=224)` (`src/data/transform.py`):
random horizontal/vertical flip, random rotation, color jitter, normalization with ImageNet mean/std.

Validation and test sets use `get_eval_transforms(image_size=224)`: resize + center crop + normalize only.

### Checkpoint Selection

The best checkpoint is saved whenever **validation AUC-ROC improves**:

```python
if val_metrics['auc'] > best_val_auc:
    best_val_auc = val_metrics['auc']
    torch.save(model.state_dict(), 'models/efficientnet_b0_l1_metadata_best.pth')
```

Best checkpoint was saved at **epoch 21** (Val AUC: **0.9167**).

### Running It

Open and run all cells in `notebooks/efficientnet/06.efficientnet_b0_l1_metadata_tta.ipynb`.  
Training takes approximately 30–45 minutes on a single GPU.

---

## Step 2 — Train MobileNetV3-Large with Metadata

**Reference notebook:** `notebooks/mobilenet/01.mobilenet_v3_metadata_tta.ipynb`  
**Output checkpoint:** `models/01.mobilenet_v3_metadata_best.pth`

### Architecture

The model is defined in PyTorch as `MobileNetV3LargeWithMetadata` in `src/models/mobilenet.py`. It follows the same fusion design:

- **Image branch:** MobileNetV3-Large pretrained on ImageNet (`features` + `avgpool`) → **960-dim** feature vector
- **Metadata branch:** `Linear(17, 32) → ReLU` → **32-dim** feature vector
- **Fusion head:** `cat(960, 32)` → `Dropout(0.5)` → `Linear(992, 1)`

Unlike EfficientNet-B0, the **full backbone is unfrozen** from the start:

```python
model = MobileNetV3LargeWithMetadata(metadata_dim=17, num_classes=1, freeze_backbone=False, dropout=0.5)
```

Trainable parameters: **~3M / 3M total** (entire model).

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR — backbone (`features` + `avgpool`) | 1e-4 |
| LR — metadata encoder + head | 1e-3 |
| Weight decay | 1e-3 |
| L1 regularization (λ) | 1e-3 |
| L2 regularization (λ) | 1e-3 |
| Dropout | 0.5 |
| Loss | `BCEWithLogitsLoss(pos_weight=8.11)` |
| Scheduler | `CosineAnnealingLR(T_max=30)` |
| Epochs | 30 |
| Batch size | 32 |
| Image size | 224×224 |
| Seed | 42 |

Training augmentation and checkpoint selection strategy are identical to Step 1.

Best checkpoint was saved at **epoch 16** (Val AUC: **0.9188**).

### Running It

Open and run all cells in `notebooks/mobilenet/01.mobilenet_v3_metadata_tta.ipynb`.  
Training takes approximately 30–45 minutes on a single GPU.

---

## Step 3 — Ensemble Evaluation

**Reference notebook:** `notebooks/Group16.ipynb`  
**No training required** — both checkpoints are loaded directly.

### Ensemble Strategy

Each model independently runs **8× deterministic TTA** per sample. The TTA augmentations are:

1. Identity (no augmentation)
2. Horizontal flip
3. Vertical flip
4. Horizontal + vertical flip
5. Rotate 90°
6. Rotate 180°
7. Rotate 270°
8. Rotate 45°

The **mean probability** across all 8 augmented views is computed for each model. The ensemble prediction is then the mean of the two per-model probabilities:

```
p_ensemble = (p_efficientnet + p_mobilenet) / 2
```

### Threshold Tuning

A threshold sweep from 0.01 to 0.90 is run on the **validation set** to find the value maximising F2. The best threshold (0.50) is then applied to the test set.

### Expected Results

| Metric | Value |
|---|---|
| Threshold | 0.50 |
| Val F2 | 0.7023 |
| Test AUC-ROC | **0.9235** |
| Test Balanced Accuracy | ~0.84 |
| Test F2 | 0.6781 |

---

## Summary of Steps

| Step | Action | Notebook | Output |
|---|---|---|---|
| 1 | Train EfficientNet-B0 + metadata, 30 epochs | `efficientnet/06.efficientnet_b0_l1_metadata_tta.ipynb` | `models/efficientnet_b0_l1_metadata_best.pth` |
| 2 | Train MobileNetV3-Large + metadata, 30 epochs | `mobilenet/01.mobilenet_v3_metadata_tta.ipynb` | `models/01.mobilenet_v3_metadata_best.pth` |
| 3 | Load both checkpoints, ensemble TTA, evaluate | `notebooks/Group16.ipynb` | AUC 0.9235, F2 0.6781 |
