import torch.nn as nn
import torchvision.models as models


class DenseNetPretrainedWeighted(nn.Module):
    def __init__(self, num_classes: int = 1, freeze_features: bool = True):
        super().__init__()

        self.model = models.densenet121(pretrained=True)

        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
