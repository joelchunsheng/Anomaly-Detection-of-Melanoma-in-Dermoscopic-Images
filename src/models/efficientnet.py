import torch
import torch.nn as nn
from torchvision import models

_EFFICIENTNET_CONFIGS = {
    'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
    'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    'b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
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


class EfficientNetB0WithMetadata(nn.Module):
    """EfficientNet-B0 backbone fused with a patient metadata encoder.

    Architecture:
        - EfficientNet-B0 features + avgpool → 1280-dim image features
        - Metadata MLP: Linear(metadata_dim, 32) → ReLU → 32-dim meta features
        - Fusion head: Dropout → Linear(1280 + 32, num_classes)

    The backbone is frozen by default. Unfreeze blocks in the notebook after
    instantiation, e.g. to unfreeze the last 6 blocks:
        for block in list(model.features)[-6:]:
            for param in block.parameters():
                param.requires_grad = True

    model.features has 9 children:
        [0] stem conv, [1-7] MBConv blocks, [8] head conv
    """

    def __init__(
        self,
        metadata_dim: int = 17,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_backbone:
            for param in efficientnet.parameters():
                param.requires_grad = False

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
        )

        in_features = efficientnet.classifier[1].in_features  # 1280 for B0
        fusion_dim = in_features + 32

        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.avgpool(self.features(images)).flatten(1)  # (B, 1280)
        meta_feat = self.meta_encoder(metadata)                     # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)


class EfficientNetB4WithMetadata(nn.Module):
    """EfficientNet-B4 backbone fused with a patient metadata encoder.

    Architecture:
        - EfficientNet-B4 features + avgpool → 1792-dim image features
        - Metadata MLP: Linear(metadata_dim, 32) → ReLU → 32-dim meta features
        - Fusion head: Dropout → Linear(1792 + 32, num_classes)

    Native input resolution: 380×380.
    The backbone is frozen by default. Unfreeze blocks in the notebook after
    instantiation, e.g. to unfreeze the last 6 blocks:
        for block in list(model.features)[-6:]:
            for param in block.parameters():
                param.requires_grad = True

    model.features has 9 children:
        [0] stem conv, [1-7] MBConv blocks, [8] head conv
    """

    def __init__(
        self,
        metadata_dim: int = 17,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

        if freeze_backbone:
            for param in efficientnet.parameters():
                param.requires_grad = False

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
        )

        in_features = efficientnet.classifier[1].in_features  # 1792 for B4
        fusion_dim = in_features + 32

        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.avgpool(self.features(images)).flatten(1)  # (B, 1792)
        meta_feat = self.meta_encoder(metadata)                     # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)
