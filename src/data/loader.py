from pathlib import Path
from typing import Any

import numpy as np
import torch
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

        sample_tensor = torch.from_numpy(sample).unsqueeze(0).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample_tensor, label_tensor


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = DigitSumDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
