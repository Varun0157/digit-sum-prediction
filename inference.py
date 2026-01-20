"""
Inference script for digit sum prediction.

Usage:
    from inference import predict
    predictions = predict(images)  # images: [N, 40, 168] or [N, 1, 40, 168]
"""

import numpy as np
import torch

from src.model.multihead import MultiHeadResNet

# model configuration
CHECKPOINT_PATH = "checkpoints/multihead_resnet_best_aug_full.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("note: running on cpu")

# load model
model = MultiHeadResNet(
    num_digits=4,
    width_multiplier=1.0,
    kernel_size=7,
    dropout=0.3,
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"Loaded model from {CHECKPOINT_PATH}")
print(f"Using device: {DEVICE}")


def predict(x):
    """
    Predict sum of digits for input images.

    Args:
        x: numpy.ndarray or torch.Tensor of shape [N, 40, 168] or [N, 1, 40, 168]
           Images should be uint8 (0-255) or float (0-1)

    Returns:
        numpy.ndarray of shape [N] with predicted sums (0-36)
    """
    # Convert to tensor if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    # Add channel dimension if needed: [N, 40, 168] -> [N, 1, 40, 168]
    if x.ndim == 3:
        x = x.unsqueeze(1)

    # Normalize to [0, 1] if needed (assumes uint8 input)
    if x.max() > 1.0:
        x = x / 255.0

    x = x.to(DEVICE)

    with torch.no_grad():
        # Get digit predictions from 4 heads
        digit_logits = model(x)  # List of 4 tensors, each [N, 10]

        # Get predicted digit for each head
        digit_preds = [logits.argmax(dim=1) for logits in digit_logits]

        # Sum the 4 digits to get final prediction
        sum_preds = torch.stack(digit_preds, dim=1).sum(dim=1)

    return sum_preds.cpu().numpy()


def evaluate(data_dir: str = "data/test", batch_size: int = 128):
    """
    Evaluate model on test set.

    Args:
        data_dir: path to directory containing data*.npy and label*.npy files
        batch_size: batch size for inference
    """
    from glob import glob

    data_files = sorted(glob(f"{data_dir}/data*.npy"))
    label_files = sorted(glob(f"{data_dir}/label*.npy"))

    all_samples = np.concatenate([np.load(f) for f in data_files])
    all_labels = np.concatenate([np.load(f) for f in label_files])

    print(f"Loaded {len(all_samples)} samples from {data_dir}")

    all_preds = []
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i : i + batch_size]
        preds = predict(batch)
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    accuracy = (all_preds == all_labels).mean()
    mae = np.abs(all_preds - all_labels).mean()

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"MAE: {mae:.3f}")

    return accuracy, mae


if __name__ == "__main__":
    evaluate()
