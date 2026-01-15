from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


class DigitSumDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

        samples_path = self.data_dir / "samples.npy"
        labels_path = self.data_dir / "labels.npy"

        self.samples = np.load(samples_path)
        self.labels = np.load(labels_path)

        assert len(self.samples) == len(self.labels), (
            f"Samples and labels length mismatch: {len(self.samples)} vs {len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        assert sample.ndim == 2, f"Expected 2D grayscale image, got {sample.ndim}D"

        sample_tensor = torch.from_numpy(sample).unsqueeze(0).float() / 255.0
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample_tensor, label_tensor

    def get_class_weights(
        self, weight_range: tuple[float, float] | None = (1.0, 5.0)
    ) -> torch.Tensor:
        unique_classes = np.unique(self.labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=self.labels,
        )

        if weight_range is not None:
            min_new, max_new = weight_range
            min_weight, max_weight = weights.min(), weights.max()
            weights = (
                ((weights - min_weight) / (max_weight - min_weight))
                * (max_new - min_new)
            ) + min_new

        num_classes = int(self.labels.max()) + 1
        weight_tensor = torch.zeros(num_classes, dtype=torch.float32)

        unique_classes_long = torch.from_numpy(unique_classes.astype(np.int64))
        weight_tensor[unique_classes_long] = torch.from_numpy(weights).float()
        return weight_tensor


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
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
