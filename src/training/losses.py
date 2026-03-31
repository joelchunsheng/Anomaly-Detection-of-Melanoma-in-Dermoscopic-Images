import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for use with raw logits.
    Drop-in replacement for nn.BCEWithLogitsLoss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weight for the positive class. Default 0.75.
        gamma (float): Focusing parameter — down-weights easy examples. Default 2.0.
        reduction (str): 'mean' or 'sum'. Default 'mean'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # p_t: probability assigned to the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t: alpha for positives, (1 - alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - p_t) ** self.gamma * (-torch.log(p_t))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
