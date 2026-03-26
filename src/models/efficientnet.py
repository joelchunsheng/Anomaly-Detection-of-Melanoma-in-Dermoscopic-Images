import torch.nn as nn
from torchvision import models

_EFFICIENTNET_CONFIGS = {
    'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
    'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
}


def get_efficientnet(variant: str = 'b3', num_classes: int = 1, freeze_backbone: bool = False, dropout: float = 0.0):
    model_fn, weights = _EFFICIENTNET_CONFIGS[variant]
    model = model_fn(weights=weights)

    if freeze_backbone:
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
