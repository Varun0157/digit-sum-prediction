"""
Multi-head ResNet-inspired models for digit-wise prediction.

All variants share a ResNetBackbone encoder and a MultiDigitBase interface.

Variants:
- MultiHeadResNet: 4 independent linear heads on pooled features (~1.2M params)
- MultiHeadSpatialAttention: per-head learned spatial attention instead of avg pool
- GRUHead: GRU decoder over 4 steps, initialized from pooled backbone features

Input: Nx1x40x168
Output: Nx4x10 (logits for each digit position)
"""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, Labels


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers and skip connection.

    Structure:
    x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
    └──────────────────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNetBackbone(nn.Module):
    """
    Shared ResNet-inspired feature extractor.

    Returns feature maps (N, out_channels, H, W) — pooling is left to the head.
    """

    def __init__(self, kernel_size: int = 7, width_multiplier: float = 1.0):
        super().__init__()

        base_channels = [32, 64, 128, 256]
        c1, c2, c3, c4 = [int(c * width_multiplier) for c in base_channels]
        self.out_channels = c4

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            1, c1, kernel_size=kernel_size, stride=2, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(c1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualBlock(c1, c2, stride=1)
        self.layer2 = ResidualBlock(c2, c3, stride=2)
        self.layer3 = ResidualBlock(c3, c4, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MultiDigitBase(BaseModel):
    """
    Abstract base for multi-digit models.

    Provides default apply_criterion (per-digit cross entropy) and get_sum.
    Subclasses must implement forward(); they may override apply_criterion if needed.
    """

    num_digits: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def apply_criterion(
        self,
        logits: torch.Tensor,
        labels: Labels,
        criterion: nn.Module,
    ) -> torch.Tensor:
        assert "digits" in labels
        digits = labels["digits"]
        return sum(
            criterion(logits[:, i], digits[:, i]) for i in range(self.num_digits)
        )

    def get_sum(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=2).sum(dim=1)


class MultiHeadResNet(MultiDigitBase):
    """4 independent linear heads on globally pooled backbone features."""

    def __init__(
        self,
        num_digits: int = 4,
        dropout: float = 0.3,
        width_multiplier: float = 1.0,
        kernel_size: int = 7,
        sum_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.num_digits = num_digits
        self.sum_loss_weight = sum_loss_weight

        self.backbone = ResNetBackbone(
            kernel_size=kernel_size, width_multiplier=width_multiplier
        )
        c4 = self.backbone.out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if width_multiplier > 1.0:
            hidden_dim = int(c4 * 0.4)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(c4, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(hidden_dim, 10),
                    )
                    for _ in range(num_digits)
                ]
            )
        else:
            self.heads = nn.ModuleList([nn.Linear(c4, 10) for _ in range(num_digits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # (N, 256, 3, 11)
        x = self.avgpool(x)  # (N, 256, 1, 1)
        x = torch.flatten(x, 1)  # (N, 256)
        x = self.dropout(x)  # (N, 256)
        return torch.stack([head(x) for head in self.heads], dim=1)  # (N, 4, 10)

    def apply_criterion(
        self,
        logits: torch.Tensor,
        labels: Labels,
        criterion: nn.Module,
    ) -> torch.Tensor:
        loss = super().apply_criterion(logits, labels, criterion)

        if self.sum_loss_weight > 0.0:
            assert "sum" in labels
            digit_range = torch.arange(10, device=logits.device, dtype=torch.float)
            expected_sum = sum(
                (F.softmax(logits[:, i], dim=1) * digit_range).sum(dim=1)
                for i in range(self.num_digits)
            )
            loss = loss + self.sum_loss_weight * F.mse_loss(
                expected_sum, labels["sum"].float()
            )

        return loss


class MultiHeadSpatialAttention(MultiDigitBase):
    """Per-head learned spatial attention replacing global average pooling."""

    def __init__(self, num_digits: int = 4, dropout: float = 0.3):
        super().__init__()

        self.num_digits = num_digits
        self.backbone = ResNetBackbone()
        c4 = self.backbone.out_channels

        self.spatial_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c4, c4 // 4, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(c4 // 4, 1, kernel_size=1),
                )
                for _ in range(num_digits)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(c4, 10) for _ in range(num_digits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # (N, 256, 3, 11)

        N, C, H, W = x.shape
        x_flat = x.view(N, C, -1)  # (N, 256, 33)

        outputs = []
        for attn_net, head in zip(self.spatial_attention, self.heads):
            attn_weights = F.softmax(attn_net(x).view(N, -1), dim=1)  # (N, 33)
            pooled = torch.bmm(x_flat, attn_weights.unsqueeze(2)).squeeze(2)  # (N, 256)
            outputs.append(head(self.dropout(pooled)))  # (N, 10)

        return torch.stack(outputs, dim=1)  # (N, 4, 10)


class GRUHead(MultiDigitBase):
    """
    GRU-based sequential digit prediction.

    The backbone feature vector initializes the GRU hidden state. A learned
    step embedding is fed as input at each of the 4 steps, and each step's
    hidden state is classified into a digit (0-9) by a single shared head.
    """

    def __init__(self, num_digits: int = 4, dropout: float = 0.3):
        super().__init__()

        self.num_digits = num_digits
        self.backbone = ResNetBackbone()
        c4 = self.backbone.out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.step_emb = nn.Embedding(num_digits, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=c4, batch_first=False)
        self.head = nn.Linear(c4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        x = self.backbone(x)  # (N, 256, 3, 11)
        x = self.avgpool(x)  # (N, 256, 1, 1)
        x = torch.flatten(x, 1)  # (N, 256)
        x = self.dropout(x)  # (N, 256)

        h0 = x.unsqueeze(0)  # (1, N, 256)

        steps = torch.arange(self.num_digits, device=x.device)
        embs = self.step_emb(steps).unsqueeze(1).expand(-1, N, -1)  # (4, N, 32)

        out, _ = self.gru(embs, h0)  # (4, N, 256)
        logits = self.head(out).permute(1, 0, 2)  # (N, 4, 10)
        return logits
