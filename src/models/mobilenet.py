import torch
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


class MobileNetV3LargeWithMetadata(nn.Module):
    """MobileNetV3-Large backbone fused with a patient metadata encoder.

    Architecture:
        - MobileNetV3-Large features + avgpool → 960-dim image features
        - Metadata MLP: Linear(metadata_dim, 32) → ReLU → 32-dim meta features
        - Fusion head: Dropout → Linear(960 + 32, num_classes)

    The backbone is fully unfrozen by default.

    model.features has 16 children:
        [0] stem conv, [1-15] InvertedResidual / SE blocks
    """

    def __init__(
        self,
        metadata_dim: int = 17,
        num_classes: int = 1,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        if freeze_backbone:
            for param in mobilenet.parameters():
                param.requires_grad = False

        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
        )

        # MobileNetV3-Large: features output channels = 960
        img_feature_dim = 960
        fusion_dim = img_feature_dim + 32

        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.avgpool(self.features(images)).flatten(1)  # (B, 960)
        meta_feat = self.meta_encoder(metadata)                     # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)
