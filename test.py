"""Inference script for digit sum prediction."""

import numpy as np
import torch
from glob import glob

from src.model.base import BaseModel
from src.model.baseline import SimpleCNN
from src.model.multihead import GRUHead, MultiHeadResNet


def load_multihead(
    device: torch.device,
    checkpoint: str = "checkpoints/MultiHeadResNet_k7_aug.pth",
) -> MultiHeadResNet:
    model = MultiHeadResNet(num_digits=4, width_multiplier=1.0, kernel_size=7, dropout=0.3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def load_gru(
    device: torch.device,
    checkpoint: str = "checkpoints/GRUHead_aug.pth",
) -> GRUHead:
    model = GRUHead(num_digits=4, dropout=0.3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def load_baseline(
    device: torch.device,
    checkpoint: str = "checkpoints/SimpleCNN_k7_avg_unweighted.pth",
) -> SimpleCNN:
    model = SimpleCNN(num_classes=37, kernel_size=7, pool_type="avg")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def predict(model: BaseModel, x: np.ndarray | torch.Tensor, device: torch.device) -> np.ndarray:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.max() > 1.0:
        x = x / 255.0

    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        preds = model.get_sum(logits)

    return preds.cpu().numpy()


def evaluate(data_dir: str = "data/test", batch_size: int = 128) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_files = sorted(glob(f"{data_dir}/data*.npy"))
    label_files = sorted(glob(f"{data_dir}/label*.npy"))

    samples = np.concatenate([np.load(f) for f in data_files])
    labels = np.concatenate([np.load(f) for f in label_files])

    print(f"Loaded {len(samples)} samples from {data_dir}\n")

    models: list[tuple[str, BaseModel]] = [
        ("Multihead ResNet", load_multihead(device)),
        ("GRU Head", load_gru(device)),
        ("Baseline SimpleCNN", load_baseline(device)),
    ]

    for name, model in models:
        preds = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            preds.append(predict(model, batch, device))
        preds = np.concatenate(preds)

        acc = (preds == labels).mean() * 100
        mae = np.abs(preds - labels).mean()
        print(f"{name}:\n  Accuracy: {acc:.2f}%\n  MAE: {mae:.3f}\n")


if __name__ == "__main__":
    evaluate()
