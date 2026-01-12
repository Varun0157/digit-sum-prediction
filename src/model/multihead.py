"""
Multi-head ResNet-inspired model for digit-wise prediction.

Architecture:
- ResNet-inspired backbone with skip connections
- 3 residual blocks (32→64→128→256 channels)
- 4 separate classification heads (one per digit position)
- Each head predicts 0-9 for its digit

Input: Nx1x40x168
Output: 4 heads, each Nx10 (logits for digits 0-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers and skip connection.

    Structure:
    x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
    └──────────────────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (1x1 conv if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = F.relu(out)

        return out


class MultiHeadResNet(nn.Module):
    """
    ResNet-inspired multi-head model for digit prediction.

    Backbone extracts shared features, then 4 heads predict each digit.
    """

    def __init__(self, num_digits=4, dropout=0.3):
        super().__init__()

        self.num_digits = num_digits

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = ResidualBlock(32, 64, stride=1)    # 32→64
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 64→128
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 128→256

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Separate classification heads for each digit position
        self.heads = nn.ModuleList([
            nn.Linear(256, 10) for _ in range(num_digits)
        ])

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))  # Nx1x40x168 → Nx32x20x84
        x = self.maxpool(x)                   # Nx32x20x84 → Nx32x10x42

        x = self.layer1(x)                    # Nx32x10x42 → Nx64x10x42
        x = self.layer2(x)                    # Nx64x10x42 → Nx128x5x21
        x = self.layer3(x)                    # Nx128x5x21 → Nx256x3x11

        # Global pooling
        x = self.avgpool(x)                   # Nx256x3x11 → Nx256x1x1
        x = torch.flatten(x, 1)               # Nx256x1x1 → Nx256

        # Apply dropout
        x = self.dropout(x)

        # Multi-head prediction
        outputs = [head(x) for head in self.heads]  # 4 x [Nx10]

        return outputs  # Returns list of 4 tensors, each Nx10
