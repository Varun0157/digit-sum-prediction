from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src.model.base import Labels


class Batch(TypedDict):
    """Batch returned by the dataloader."""

    image: torch.Tensor  # (N, 1, 40, 168)
    labels: Labels


class DigitSumDataset(Dataset[Batch]):
    """Dataset for digit sum prediction.

    Loads images and labels from a directory containing:
    - samples.npy: (N, 40, 168) uint8 images
    - sum_labels.npy: (N,) sum labels (0-36)
    - digit_labels.npy: (N, 4) per-digit labels (optional)
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

        self.samples = np.load(self.data_dir / "samples.npy")
        self.sum_labels = np.load(self.data_dir / "sum_labels.npy")

        digit_labels_path = self.data_dir / "digit_labels.npy"
        self.digit_labels = (
            np.load(digit_labels_path) if digit_labels_path.exists() else None
        )

        assert len(self.samples) == len(self.sum_labels)
        if self.digit_labels is not None:
            assert len(self.samples) == len(self.digit_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Batch:
        image = torch.from_numpy(self.samples[idx]).unsqueeze(0).float() / 255.0
        labels: Labels = {"sum": torch.tensor(self.sum_labels[idx], dtype=torch.long)}

        if self.digit_labels is not None:
            labels["digits"] = torch.tensor(self.digit_labels[idx], dtype=torch.long)

        return {"image": image, "labels": labels}

    def get_class_weights(
        self, weight_range: tuple[float, float] | None = (1.0, 5.0)
    ) -> torch.Tensor:
        """Compute balanced class weights for sum labels."""
        unique_classes = np.unique(self.sum_labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=self.sum_labels,
        )

        if weight_range is not None:
            min_new, max_new = weight_range
            min_weight, max_weight = weights.min(), weights.max()
            weights = (
                ((weights - min_weight) / (max_weight - min_weight))
                * (max_new - min_new)
            ) + min_new

        num_classes = int(self.sum_labels.max()) + 1
        weight_tensor = torch.zeros(num_classes, dtype=torch.float32)
        weight_tensor[torch.from_numpy(unique_classes.astype(np.int64))] = (
            torch.from_numpy(weights).float()
        )
        return weight_tensor


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> tuple[DataLoader[Batch], torch.Tensor]:
    dataset = DigitSumDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    weights = dataset.get_class_weights()
    return loader, weights
