import torch.nn as nn
from torchvision import models


def get_vit(
    num_classes: int = 1,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    if dropout > 0.0:
        model.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.heads = nn.Sequential(
            nn.Linear(in_features, num_classes),
        )

    return model
