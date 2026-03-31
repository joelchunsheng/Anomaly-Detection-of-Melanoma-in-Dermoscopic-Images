import torch.nn as nn
import torchvision.models as models


class DenseNetDropoutWeighted(nn.Module):
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5):
        super().__init__()

        self.model = models.densenet121(pretrained=False)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
