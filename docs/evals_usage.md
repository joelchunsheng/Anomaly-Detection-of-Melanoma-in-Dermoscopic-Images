# Evaluation Utility — Migration Guide

This document instructs how to migrate a training notebook to use the shared evaluation utilities in `src/utils/evaluation.py`. Follow each step exactly.

---

## What changed and why

All notebooks previously duplicated the same evaluation code (metric collection, curve plotting, threshold tuning, confusion matrix) with minor inconsistencies. This utility centralises that logic into four importable functions.

Raw accuracy has also been replaced with **balanced accuracy** across all tracked metrics, because the HAM10000 dataset is class-imbalanced (melanoma is the minority class) and raw accuracy is misleading in that setting. Balanced accuracy is the average recall per class.

---

## Available functions

All four functions are exported from `src/utils`:

| Function | Purpose |
|---|---|
| `plot_training_curves(train_history, val_history)` | Plot Loss, Balanced Accuracy, Recall, F2 in a 2×2 grid |
| `get_predictions(model, dataloader, device, threshold=0.5)` | Run inference, return `(labels, probs, preds)` |
| `find_best_threshold(model, val_loader, device)` | Sweep thresholds 0.01–0.90, return `(best_threshold, best_f2)` |
| `evaluate_model(model, test_loader, device, threshold=0.5)` | Print report + AUC-ROC + confusion matrix plot |

---

## Step-by-step migration

### Step 1 — Add the import

In the notebook's first code cell, alongside the existing imports, add:

```python
from src.utils import plot_training_curves, find_best_threshold, evaluate_model
```

No other imports need to change. The existing `from src.training.trainer import train_one_epoch, validate_one_epoch` stays as-is.

---

### Step 2 — Update the training loop

The training loop currently appends to several separate lists and references the `'accuracy'` key. Replace the entire training loop with the pattern below.

**Remove** all of these list declarations:
```python
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_recalls, val_recalls = [], []
train_f2s, val_f2s = [], []
```

**Replace with:**
```python
train_history, val_history = [], []
```

Inside the loop, **remove** all individual `.append()` calls and **replace with:**
```python
train_history.append(train_metrics)
val_history.append(val_metrics)
```

Update the `print` statement to use `'balanced_accuracy'` instead of `'accuracy'`:
```python
print(
    f"Epoch [{epoch+1}/{num_epochs}] | "
    f"Train Loss: {train_metrics['loss']:.4f}, Bal Acc: {train_metrics['balanced_accuracy']:.4f}, "
    f"Recall: {train_metrics['recall']:.4f}, F2: {train_metrics['f2']:.4f} | "
    f"Val Loss: {val_metrics['loss']:.4f}, Bal Acc: {val_metrics['balanced_accuracy']:.4f}, "
    f"Recall: {val_metrics['recall']:.4f}, F2: {val_metrics['f2']:.4f}"
)
```

The best-model checkpoint logic does not change:
```python
if val_metrics['f2'] > best_val_f2:
    best_val_f2 = val_metrics['f2']
    torch.save(model.state_dict(), 'models/<model_name>_best.pth')
    print('Saved best model at epoch', epoch+1)
```

> `train_one_epoch` and `validate_one_epoch` now return `balanced_accuracy` instead of `accuracy`. Do not reference `'accuracy'` anywhere — it no longer exists in the returned dict.

---

### Step 3 — Replace the plotting cells

**Remove** all cells that individually plot loss, accuracy, recall, and F2 (typically 4 separate `plt.figure()` blocks).

**Replace with a single cell:**
```python
plot_training_curves(train_history, val_history)
```

This produces one figure with four subplots: Loss, Balanced Accuracy, Recall (Melanoma), F2 (Melanoma).

---

### Step 4 — Replace threshold tuning

**Remove** the existing threshold tuning block. It typically looks like:

```python
# REMOVE THIS ENTIRE BLOCK
val_probs = []
val_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        ...
thresholds = np.arange(0.01, 0.9, 0.01)
f2_scores = [fbeta_score(...) for t in thresholds]
best_threshold = thresholds[np.argmax(f2_scores)]
print(...)
```

**Replace with:**
```python
model.load_state_dict(torch.load('models/<model_name>_best.pth', map_location=device))
best_threshold, best_f2 = find_best_threshold(model, val_loader, device)
```

The function prints the best threshold and its F2 score automatically.

---

### Step 5 — Replace the test evaluation cells

**Remove** all cells that manually run inference on the test set and display results. This typically spans several cells covering:
- A `torch.no_grad()` inference loop over `test_loader`
- `confusion_matrix(...)` and `print(cm)`
- `classification_report(...)` print
- `ConfusionMatrixDisplay(...).plot(...)`

**Replace all of them with a single cell:**
```python
evaluate_model(model, test_loader, device, threshold=best_threshold)
```

This prints:
- The threshold used
- AUC-ROC score
- Balanced accuracy
- F2 score
- Full classification report (precision, recall, F1, support per class)
- Confusion matrix plot labelled `["Non-Melanoma", "Melanoma"]`

---

## Complete example — what a migrated notebook looks like

```python
# Cell 1 — Imports
import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / 'src').exists())
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloaders
from src.models.resnet import get_resnet
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.utils import plot_training_curves, find_best_threshold, evaluate_model
```

```python
# Cell 2 — Training loop
best_val_f2 = 0.0
train_history, val_history = [], []

for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics   = validate_one_epoch(model, val_loader, criterion, device)

    train_history.append(train_metrics)
    val_history.append(val_metrics)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_metrics['loss']:.4f}, Bal Acc: {train_metrics['balanced_accuracy']:.4f}, "
        f"Recall: {train_metrics['recall']:.4f}, F2: {train_metrics['f2']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f}, Bal Acc: {val_metrics['balanced_accuracy']:.4f}, "
        f"Recall: {val_metrics['recall']:.4f}, F2: {val_metrics['f2']:.4f}"
    )

    if val_metrics['f2'] > best_val_f2:
        best_val_f2 = val_metrics['f2']
        torch.save(model.state_dict(), 'models/<model_name>_best.pth')
        print('Saved best model at epoch', epoch+1)
```

```python
# Cell 3 — Training curves
plot_training_curves(train_history, val_history)
```

```python
# Cell 4 — Threshold tuning
model.load_state_dict(torch.load('models/<model_name>_best.pth', map_location=device))
best_threshold, best_f2 = find_best_threshold(model, val_loader, device)
```

```python
# Cell 5 — Test evaluation
evaluate_model(model, test_loader, device, threshold=best_threshold)
```

---

## Quick reference — key changes summary

| Area | Before | After |
|---|---|---|
| Import | — | `from src.utils import plot_training_curves, find_best_threshold, evaluate_model` |
| Training history | 8 separate lists | `train_history`, `val_history` (lists of dicts) |
| Metric key | `'accuracy'` | `'balanced_accuracy'` |
| Curve plots | 4 separate `plt.figure()` cells | `plot_training_curves(train_history, val_history)` |
| Threshold tuning | ~15-line manual block | `find_best_threshold(model, val_loader, device)` |
| Test evaluation | 4–5 manual cells | `evaluate_model(model, test_loader, device, threshold=best_threshold)` |
