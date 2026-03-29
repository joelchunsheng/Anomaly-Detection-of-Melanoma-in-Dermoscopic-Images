import torch.nn as nn
import torchvision.models as models


class DenseNetBaseline(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.model = models.densenet121(pretrained=False)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)