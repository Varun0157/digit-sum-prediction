"""Inference script for digit sum prediction."""

import numpy as np
import torch
from glob import glob

from src.model.multihead import MultiHeadResNet
from src.model.baseline import SimpleCNN


def get_multihead(device, checkpoint="checkpoints/multihead_resnet_best_aug_full.pth"):
    def load():
        model = MultiHeadResNet(
            num_digits=4, width_multiplier=1.0, kernel_size=7, dropout=0.3
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        return model.to(device).eval()

    def extract(logits):
        digit_preds = [l.argmax(dim=1) for l in logits]
        return torch.stack(digit_preds, dim=1).sum(dim=1)

    return load, extract


def get_baseline(device, checkpoint="checkpoints/SimpleCNN_k7_avg_unweighted.pth"):
    def load():
        model = SimpleCNN(num_classes=37, kernel_size=7, pool_type="avg")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        return model.to(device).eval()

    def extract(logits):
        return logits.argmax(dim=1)

    return load, extract


def predict(model, x, extract_fn, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.max() > 1.0:
        x = x / 255.0

    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        preds = extract_fn(logits)

    return preds.cpu().numpy()


def evaluate(data_dir="data/test", batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_files = sorted(glob(f"{data_dir}/data*.npy"))
    label_files = sorted(glob(f"{data_dir}/label*.npy"))

    samples = np.concatenate([np.load(f) for f in data_files])
    labels = np.concatenate([np.load(f) for f in label_files])

    print(f"Loaded {len(samples)} samples from {data_dir}\n")

    configs = [
        ("Multihead ResNet", get_multihead(device)),
        ("Baseline SimpleCNN", get_baseline(device)),
    ]

    for name, (load_fn, extract_fn) in configs:
        model = load_fn()

        preds = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            preds.append(predict(model, batch, extract_fn, device))
        preds = np.concatenate(preds)

        acc = (preds == labels).mean() * 100
        mae = np.abs(preds - labels).mean()
        print(f"{name}:\n  Accuracy: {acc:.2f}%\n  MAE: {mae:.3f}\n")


if __name__ == "__main__":
    evaluate()
