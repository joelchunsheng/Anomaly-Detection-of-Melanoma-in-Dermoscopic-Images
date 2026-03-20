import torch.nn as nn
from torchvision import models


def get_resnet(num_classes: int = 1, freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
