"""Base model class for digit sum prediction."""

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
import torch.nn as nn


class Labels(TypedDict, total=False):
    """Labels dictionary passed to models."""

    sum: torch.Tensor  # (N,) - sum labels (0-36)
    digits: torch.Tensor  # (N, 4) - per-digit labels (each 0-9)


class BaseModel(nn.Module, ABC):
    """Abstract base class for digit sum prediction models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images, shape (N, 1, 40, 168)

        Returns:
            Logits - shape depends on model architecture
        """
        pass

    @abstractmethod
    def apply_criterion(
        self,
        logits: torch.Tensor,
        labels: Labels,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Output from forward()
            labels: Dict containing label tensors
            criterion: Loss function (e.g., CrossEntropyLoss)

        Returns:
            Scalar loss tensor
        """
        pass

    @abstractmethod
    def get_sum(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract sum predictions from logits.

        Args:
            logits: Output from forward()

        Returns:
            Predicted sums, shape (N,)
        """
        pass
