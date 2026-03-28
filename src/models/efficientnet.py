import torch.nn as nn
from torchvision import models

_EFFICIENTNET_CONFIGS = {
    'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
    'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
}


def get_efficientnet(variant: str = 'b3', num_classes: int = 1, freeze_backbone: bool = False, unfreeze_last_n_blocks: int = 0, dropout: float = 0.0):
    """
    unfreeze_last_n_blocks: if > 0, freeze entire backbone then unfreeze only the
    last N children of model.features. model.features has 9 children:
      [0] stem conv, [1-7] MBConv blocks, [8] head conv.
    e.g. unfreeze_last_n_blocks=3 trains only MBConv block 6, block 7, and head conv.
    If 0 (default), all backbone layers are trainable.
    """
    model_fn, weights = _EFFICIENTNET_CONFIGS[variant]
    model = model_fn(weights=weights)

    if unfreeze_last_n_blocks > 0:
        for param in model.parameters():
            param.requires_grad = False
        for block in list(model.features)[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    elif freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    if dropout > 0.0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
        )

    return model
