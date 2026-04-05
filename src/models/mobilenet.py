import torch.nn as nn
from torchvision import models


def get_mobilenet_v3_small(num_classes: int = 1, freeze_backbone: bool = False, unfreeze_last_n_blocks: int = 0, dropout: float = 0.2):
    """
    MobileNetV3-Small (~2.5M params).

    model.features has 13 children:
      [0] stem conv, [1-12] InvertedResidual / SE blocks.
    unfreeze_last_n_blocks: if > 0, freeze entire backbone then unfreeze
      only the last N children of model.features.
    If 0 (default), all backbone layers are trainable.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    if unfreeze_last_n_blocks > 0:
        for param in model.parameters():
            param.requires_grad = False
        for block in list(model.features)[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    elif freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
