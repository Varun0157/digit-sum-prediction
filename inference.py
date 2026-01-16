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


if __name__ == "__main__":
    # Quick test
    print("\nTesting inference...")
    dummy_input = np.random.randint(0, 256, (4, 40, 168), dtype=np.uint8)
    preds = predict(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predictions: {preds}")
    print(f"Predictions shape: {preds.shape}")
