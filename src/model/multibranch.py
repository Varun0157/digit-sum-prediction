import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, Labels


class ConvBranch(nn.Module):
    """
    Single convolutional branch with specified kernel size.
    Channel sizes: 1→32→64→128→128 (same as SimpleCNN).
    """

    def __init__(self, kernel_size: int, pool_type: str = "avg") -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # Nx1x40x168 -> Nx32x20x84
        x = self.pool(F.relu(self.conv2(x)))  # Nx32x20x84 -> Nx64x10x42
        x = self.pool(F.relu(self.conv3(x)))  # Nx64x10x42 -> Nx128x5x21
        x = self.pool(F.relu(self.conv4(x)))  # Nx128x5x21 -> Nx128x2x10
        x = torch.flatten(x, 1)  # Nx128x2x10 -> Nx2560
        return x


class MultiBranchCNN(BaseModel):
    """
    Multi-branch CNN that processes input through multiple parallel branches
    with different kernel sizes, then merges features via concatenation.

    Each branch has identical architecture (4 conv layers) but different receptive fields.
    Features from all branches are concatenated and passed through shared FC layers.
    """

    def __init__(
        self,
        num_classes: int = 37,
        kernel_sizes: list[int] = [5, 7, 9],
        pool_type: str = "avg",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        # Create parallel branches
        self.branches = nn.ModuleList(
            [ConvBranch(kernel_size=k, pool_type=pool_type) for k in kernel_sizes]
        )

        # Each branch outputs 2560 features (128 * 2 * 10)
        # After concatenation: 2560 * num_branches
        merged_features = 2560 * self.num_branches

        # Shared fully connected layers with dropout
        self.fc1 = nn.Linear(merged_features, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate features from all branches
        merged = torch.cat(branch_outputs, dim=1)  # Nx(2560*num_branches)

        # Shared FC layers with dropout
        x = F.relu(self.fc1(merged))  # Nx(2560*num_branches) -> Nx512
        x = self.dropout(x)
        x = self.fc2(x)  # Nx512 -> Nx37
        return x

    def apply_criterion(
        self,
        logits: torch.Tensor,
        labels: Labels,
        criterion: nn.Module,
    ) -> torch.Tensor:
        assert "sum" in labels, "MultiBranchCNN requires 'sum' labels"
        return criterion(logits, labels["sum"])

    def get_sum(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)
