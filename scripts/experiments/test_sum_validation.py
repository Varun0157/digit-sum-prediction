"""
Test end-to-end pipeline: Contour detection + MNIST classification + Sum validation.
Trains a simple MNIST model (~2 min, 99% accuracy) for digit recognition.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class SimpleMNIST(nn.Module):
    """Simple CNN for MNIST (LeNet-style)."""
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


def train_mnist():
    """Train MNIST model (takes ~2 minutes)."""
    print("Training MNIST classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)

    model = SimpleMNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(3):  # 3 epochs is enough for ~99% accuracy
        epoch_loss = 0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/3'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}: Avg loss = {epoch_loss/len(train_loader):.4f}')

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/mnist_classifier.pth')
    print("Saved to checkpoints/mnist_classifier.pth\n")

    return model


def load_mnist_model():
    """Load or train MNIST model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNIST().to(device)

    if os.path.exists('checkpoints/mnist_classifier.pth'):
        print("Loading MNIST model from checkpoints/mnist_classifier.pth")
        model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
    else:
        model = train_mnist()

    model.eval()
    return model, device


def segment_digits(img, erode_size=2):
    """Segment image into digit bounding boxes."""
    if erode_size > 0:
        kernel = np.ones((erode_size, erode_size), np.uint8)
        img = cv2.erode(img.astype(np.uint8), kernel)
    else:
        img = img.astype(np.uint8)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    bboxes = [b for b in bboxes if b[2] > 5 and b[3] > 10]  # Filter noise
    return sorted(bboxes, key=lambda b: b[0])  # Left-to-right


def classify_digit(crop, model, device):
    """Classify a digit crop."""
    # Resize to 28x28
    resized = cv2.resize(crop, (28, 28))

    # Normalize (MNIST stats)
    normalized = (resized / 255.0 - 0.1307) / 0.3081

    # To tensor
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()


def main():
    # Load data
    print("Loading dataset...")
    samples = np.load('data/processed/train/samples.npy')
    labels = np.load('data/processed/train/labels.npy')

    # Load MNIST model
    model, device = load_mnist_model()

    print(f"\nTesting on first 100 samples")
    print("="*60)

    stats = {'4_contours': 0, 'correct_sum': 0, 'total_valid': 0}
    examples = []

    for i in range(100):
        img = samples[i]
        label_sum = labels[i]

        # Segment
        bboxes = segment_digits(img, erode_size=2)

        if len(bboxes) == 4:
            stats['4_contours'] += 1
            stats['total_valid'] += 1

            # Classify each digit
            preds = [classify_digit(img[y:y+h, x:x+w], model, device) for (x, y, w, h) in bboxes]
            pred_sum = sum(preds)

            if pred_sum == label_sum:
                stats['correct_sum'] += 1
                status = '✓'
            else:
                status = '✗'

            print(f"{i:3d}: {preds} = {pred_sum:2d} (label={label_sum:2d}) {status}")

            # Save examples
            if len(examples) < 10:
                examples.append((i, img, bboxes, preds, pred_sum, label_sum))

    # Results
    print(f"\n{'='*60}")
    print(f"Segmentation: {stats['4_contours']}/100 images had exactly 4 contours")
    if stats['total_valid'] > 0:
        acc = stats['correct_sum'] / stats['total_valid'] * 100
        print(f"Sum Accuracy: {stats['correct_sum']}/{stats['total_valid']} = {acc:.1f}%")

    # Visualize
    if examples:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for idx, (i, img, bboxes, preds, pred_sum, label_sum) in enumerate(examples):
            ax = axes[idx // 5, idx % 5]
            img_vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            for (x, y, w, h) in bboxes:
                cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            ax.imshow(img_vis)
            status = '✓' if pred_sum == label_sum else '✗'
            ax.set_title(f"#{i}: {preds} = {pred_sum} (true={label_sum}) {status}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/sum_validation.png', dpi=150)
        print(f"\nVisualization saved to results/sum_validation.png")


if __name__ == '__main__':
    main()
