import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Match dimensions for skip connection when needed
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualBatchNormCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()

        # Simple stem to extract early low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Residual stages
        self.layer1 = ResidualBlock(32, 32, downsample=False)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = ResidualBlock(32, 64, downsample=False)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = ResidualBlock(64, 128, downsample=False)
        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = ResidualBlock(128, 256, downsample=False)
        self.pool4 = nn.MaxPool2d(2)

        # Global pooling keeps parameter count much smaller
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.layer4(x)
        x = self.pool4(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    

### Model Architecture Explanation (ResidualBatchNormCNN)

# This model is a deeper convolutional neural network enhanced with residual connections and batch normalization to improve training stability and feature learning.

# - **Stem Layer**
#   - The initial convolution layer extracts low-level image features (e.g., edges, textures).
#   - Batch Normalization stabilizes the feature distribution.
#   - ReLU introduces non-linearity.

# - **Residual Blocks (Feature Extraction)**
#   - Each block contains two convolution layers with BatchNorm and ReLU.
#   - A skip (shortcut) connection adds the input of the block directly to its output.
#   - This helps preserve important features and improves gradient flow, making deeper networks easier to train.
#   - Channel depth increases progressively (32 → 64 → 128 → 256), allowing the model to learn more complex patterns.

# - **Pooling Layers**
#   - MaxPooling reduces spatial dimensions after each block.
#   - This helps the model focus on higher-level features while reducing computation.

# - **Global Pooling**
#   - `AdaptiveAvgPool2d((1,1))` compresses each feature map into a single value.
#   - This replaces large fully connected layers and improves generalization.

# - **Classifier**
#   - A small fully connected network converts extracted features into the final prediction.
#   - Dropout is used to reduce overfitting.
#   - Final output is a single logit for binary classification (melanoma vs non-melanoma).

# Overall, this architecture improves upon previous CNNs by using residual connections for better training of deeper networks and adaptive pooling for more efficient feature aggregation.