import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTBinaryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained ViT-B/16 if requested
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.model = vit_b_16(weights=weights)
        else:
            self.model = vit_b_16(weights=None)

        # Optionally freeze the feature extractor first
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the classification head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, 1)

        # If backbone was frozen, allow the new head to train
        if freeze_backbone:
            for param in self.model.heads.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)