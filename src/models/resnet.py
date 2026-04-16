import torch
import torch.nn as nn
from torchvision import models


def get_resnet(num_classes: int = 1, freeze_backbone: bool = True, dropout: float = 0.0):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def get_resnet50(num_classes: int = 1, freeze_backbone: bool = True, dropout: float = 0.0):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


class ResNet18WithMetadata(nn.Module):
    """ResNet-18 backbone fused with a patient metadata encoder.

    Architecture:
        - ResNet-18 backbone (no FC) → 512-dim image features
        - Metadata MLP: Linear(metadata_dim, 32) → ReLU → 32-dim meta features
        - Fusion head: Dropout → Linear(512 + 32, num_classes)

    The backbone is fully unfrozen by default.

    backbone children indices:
        0=conv1, 1=bn1, 2=relu, 3=maxpool,
        4=layer1, 5=layer2, 6=layer3, 7=layer4, 8=avgpool
    """

    def __init__(
        self,
        metadata_dim: int = 17,
        num_classes: int = 1,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        # Strip the FC layer; keep conv → avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
        )

        fusion_dim = resnet.fc.in_features + 32  # 512 + 32
        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(images).flatten(1)   # (B, 512)
        meta_feat = self.meta_encoder(metadata)        # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)


class ResNet50WithMetadata(nn.Module):
    """ResNet-50 backbone fused with a patient metadata encoder.

    Architecture:
        - ResNet-50 backbone (no FC) → 2048-dim image features
        - Metadata MLP: Linear(metadata_dim, 32) → ReLU → 32-dim meta features
        - Fusion head: Dropout → Linear(2048 + 32, num_classes)

    The backbone is frozen by default. Unfreeze layers in the notebook after
    instantiation (e.g. `for p in model.backbone[7].parameters(): p.requires_grad = True`
    to unfreeze layer4).

    backbone children indices:
        0=conv1, 1=bn1, 2=relu, 3=maxpool,
        4=layer1, 5=layer2, 6=layer3, 7=layer4, 8=avgpool
    """

    def __init__(
        self,
        metadata_dim: int = 17,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        # Strip the FC layer; keep conv → avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
        )

        fusion_dim = resnet.fc.in_features + 32  # 2048 + 32
        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fusion_dim, num_classes),
            )
        else:
            self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(images).flatten(1)   # (B, 2048)
        meta_feat = self.meta_encoder(metadata)        # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)
