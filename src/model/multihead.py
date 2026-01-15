"""
Multi-head ResNet-inspired model for digit-wise prediction.

Architecture:
- ResNet-inspired backbone with skip connections
- 3 residual blocks
- 4 separate classification heads (one per digit position)
- Each head predicts 0-9 for its digit

Variants:
- base: 32→64→128→256 channels, 256→10 heads (~1.2M params)
- large: 40→80→160→320 channels, 320→128→10 heads (~2M params)

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

    def __init__(self, num_digits=4, dropout=0.3, width_multiplier=1.0, kernel_size=7):
        super().__init__()

        self.num_digits = num_digits
        self.width_multiplier = width_multiplier
        self.kernel_size = kernel_size

        # Calculate channel dimensions based on width multiplier
        base_channels = [32, 64, 128, 256]
        channels = [int(c * width_multiplier) for c in base_channels]
        c1, c2, c3, c4 = channels

        # Initial convolution (kernel_size configurable for ablation)
        padding = kernel_size // 2  # same padding
        self.conv1 = nn.Conv2d(1, c1, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = ResidualBlock(c1, c2, stride=1)    # c1→c2
        self.layer2 = ResidualBlock(c2, c3, stride=2)    # c2→c3
        self.layer3 = ResidualBlock(c3, c4, stride=2)    # c3→c4

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Separate classification heads for each digit position
        # For larger models, use 2-layer heads
        if width_multiplier > 1.0:
            hidden_dim = int(c4 * 0.4)  # 320 -> 128 for width_multiplier=1.25
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c4, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim, 10)
                ) for _ in range(num_digits)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Linear(c4, 10) for _ in range(num_digits)
            ])

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply dropout
        x = self.dropout(x)

        # Multi-head prediction
        outputs = [head(x) for head in self.heads]

        return outputs  # Returns list of 4 tensors, each Nx10


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-head model with per-head spatial attention.

    Each head learns to attend to its specific digit position,
    replacing global average pooling with learned attention-weighted pooling.
    """

    def __init__(self, num_digits=4, dropout=0.3):
        super().__init__()

        self.num_digits = num_digits
        c1, c2, c3, c4 = 32, 64, 128, 256

        # Initial convolution
        self.conv1 = nn.Conv2d(1, c1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = ResidualBlock(c1, c2, stride=1)
        self.layer2 = ResidualBlock(c2, c3, stride=2)
        self.layer3 = ResidualBlock(c3, c4, stride=2)

        # Per-head spatial attention
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c4, c4 // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(c4 // 4, 1, kernel_size=1)
            ) for _ in range(num_digits)
        ])

        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.heads = nn.ModuleList([
            nn.Linear(c4, 10) for _ in range(num_digits)
        ])

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x: (N, 256, H, W) where H~3, W~11

        N, C, H, W = x.shape
        x_flat = x.view(N, C, -1)  # (N, 256, H*W)

        outputs = []
        for attn_net, head in zip(self.spatial_attention, self.heads):
            # Compute attention map
            attn_logits = attn_net(x).view(N, -1)  # (N, H*W)
            attn_weights = F.softmax(attn_logits, dim=1)  # (N, H*W)

            # Attention-weighted pooling
            pooled = torch.bmm(x_flat, attn_weights.unsqueeze(2)).squeeze(2)  # (N, 256)
            pooled = self.dropout(pooled)
            outputs.append(head(pooled))

        return outputs
