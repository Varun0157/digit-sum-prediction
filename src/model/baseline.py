import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, Labels


class SimpleCNN(BaseModel):
    """
    Simple baseline CNN for digit sum prediction.
    ref: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(
        self, num_classes: int = 37, kernel_size: int = 3, pool_type: str = "max"
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=kernel_size, padding=kernel_size // 2
        )

        match pool_type:
            case "max":
                self.pool = nn.MaxPool2d(2, 2)
            case "avg":
                self.pool = nn.AvgPool2d(2, 2)
            case _:
                raise ValueError(f"Unknown pool_type: {pool_type}")

        self.fc1 = nn.Linear(128 * 2 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # Nx1x40x168 -> Nx32x20x84
        x = self.pool(F.relu(self.conv2(x)))  # Nx32x20x84 -> Nx64x10x42
        x = self.pool(F.relu(self.conv3(x)))  # Nx64x10x42 -> Nx128x5x21
        x = self.pool(F.relu(self.conv4(x)))  # Nx128x5x21 -> Nx128x2x10
        x = torch.flatten(x, 1)  # Nx128x2x10 -> Nx2560
        x = F.relu(self.fc1(x))  # Nx2560 -> Nx256
        x = self.fc2(x)  # Nx256 -> Nx37
        return x

    def apply_criterion(
        self,
        logits: torch.Tensor,
        labels: Labels,
        criterion: nn.Module,
    ) -> torch.Tensor:
        assert "sum" in labels, "SimpleCNN requires 'sum' labels"
        return criterion(logits, labels["sum"])

    def get_sum(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)


class SimpleMNIST(nn.Module):
    """Simple CNN for MNIST digit classification (LeNet-style)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
