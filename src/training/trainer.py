import torch
from sklearn.metrics import balanced_accuracy_score, recall_score, fbeta_score
from tqdm import tqdm

from .losses import l1_penalty, l2_penalty


def train_one_epoch(model, dataloader, criterion, optimizer, device, l1_lambda=0.0, l2_lambda=0.0):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc='Train', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)  # BCE loss

        running_loss += loss.item() * images.size(0)

        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_penalty(model) 
        if l2_lambda > 0:
            loss = loss + l2_lambda * l2_penalty(model)

        loss.backward()
        optimizer.step()


        preds = (torch.sigmoid(outputs) >= 0.5).float()
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.squeeze(1).cpu().int().tolist())
        all_preds.extend(preds.squeeze(1).cpu().int().tolist())

        pbar.set_postfix(loss=f'{running_loss / total_samples:.4f}')

    epoch_loss      = running_loss / total_samples
    epoch_bal_acc   = balanced_accuracy_score(all_labels, all_preds)
    epoch_recall    = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    epoch_f2        = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)

    return {
        "loss":               epoch_loss,
        "balanced_accuracy":  epoch_bal_acc,
        "recall":             epoch_recall,
        "f2":                 epoch_f2,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc='Val', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) >= 0.5).float()
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.squeeze(1).cpu().int().tolist())
        all_preds.extend(preds.squeeze(1).cpu().int().tolist())

        pbar.set_postfix(loss=f'{running_loss / total_samples:.4f}')

    epoch_loss      = running_loss / total_samples
    epoch_bal_acc   = balanced_accuracy_score(all_labels, all_preds)
    epoch_recall    = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    epoch_f2        = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)

    return {
        "loss":               epoch_loss,
        "balanced_accuracy":  epoch_bal_acc,
        "recall":             epoch_recall,
        "f2":                 epoch_f2,
    }