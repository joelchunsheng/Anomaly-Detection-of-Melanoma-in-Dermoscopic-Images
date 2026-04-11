import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)


CLASS_NAMES = ["Non-Melanoma", "Melanoma"]


def plot_training_curves(train_history, val_history):
    """
    Plot Loss, Balanced Accuracy, Recall, and F2 curves from training history.

    Args:
        train_history: list of dicts with keys {loss, balanced_accuracy, recall, f2}
        val_history:   list of dicts with same keys
    """
    epochs = range(1, len(train_history) + 1)
    metrics = [
        ("loss",               "Loss"),
        ("balanced_accuracy",  "Balanced Accuracy"),
        ("recall",             "Recall (Melanoma)"),
        ("f2",                 "F2 Score (Melanoma)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(epochs, [m[key] for m in train_history], label="Train")
        ax.plot(epochs, [m[key] for m in val_history],   label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    plt.show()


@torch.no_grad()
def get_predictions(model, dataloader, device, threshold=0.5):
    """
    Run model inference over a dataloader.

    Returns:
        labels: np.ndarray of ground-truth labels
        probs:  np.ndarray of sigmoid probabilities
        preds:  np.ndarray of binary predictions at the given threshold
    """
    model.eval()

    all_labels = []
    all_probs  = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= threshold).astype(int)

    return labels, probs, preds


def find_best_threshold(model, val_loader, device):
    """
    Sweep thresholds 0.01–0.90 to find the value that maximises F2 on the
    validation set.

    Returns:
        best_threshold: float
        best_f2:        float
    """
    labels, probs, _ = get_predictions(model, val_loader, device, threshold=0.5)

    thresholds = np.arange(0.01, 0.90, 0.01)
    f2_scores = [
        fbeta_score(labels, (probs >= t).astype(int), beta=2, pos_label=1, zero_division=0)
        for t in thresholds
    ]

    best_idx       = int(np.argmax(f2_scores))
    best_threshold = float(thresholds[best_idx])
    best_f2        = float(f2_scores[best_idx])

    print(f"Best threshold: {best_threshold:.2f} | Val F2: {best_f2:.4f}")
    return best_threshold, best_f2


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Full evaluation on a test set: classification report, AUC-ROC, F2 score,
    and confusion matrix plot.

    Args:
        model:       trained PyTorch model
        test_loader: DataLoader for the test set
        device:      torch.device
        threshold:   decision threshold (use output of find_best_threshold)
    """
    labels, probs, preds = get_predictions(model, test_loader, device, threshold)

    auc = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    f2 = fbeta_score(labels, preds, beta=2, pos_label=1, zero_division=0)

    print(f"Threshold:          {threshold:.2f}")
    print(f"AUC-ROC:            {auc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"F2 Score:           {f2:.4f}")
    print()
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(labels, probs)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Model").plot(ax=axes[1])
    axes[1].set_title("ROC Curve")

    fig.tight_layout()
    plt.show()
