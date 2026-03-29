import torch.nn as nn
import torchvision.models as models


class DenseNetBatchNormWeighted(nn.Module):
    """
    DenseNet121 with BatchNorm1d added to classifier head.
    Class weighting (pos_weight) applied to BCEWithLogitsLoss in the notebook.
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.model = models.densenet121(pretrained=False)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
