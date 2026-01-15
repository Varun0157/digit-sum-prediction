"""
Fine-tune MNIST classifier on combined dataset:
- Manual labels (2,000 crops from hard cases)
- Pseudo labels (61,592 crops from easy cases)

Usage:
    python scripts/finetune_mnist.py --epochs 10 --lr 1e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.baseline import SimpleMNIST


class DigitCropDataset(Dataset):
    """Dataset from digit crop images with labels in filenames."""

    def __init__(self, crop_dirs, transform=None):
        """
        Args:
            crop_dirs: List of directories containing digit crops
            transform: Optional transform to apply
        """
        self.samples = []
        self.transform = transform

        # Collect all digit crops from all directories
        for crop_dir in crop_dirs:
            if not os.path.exists(crop_dir):
                print(f"Warning: {crop_dir} not found, skipping")
                continue

            for filename in os.listdir(crop_dir):
                if not filename.endswith('.png'):
                    continue

                # Parse label from filename: img00000_digit0_label7.png
                try:
                    label = int(filename.split('_label')[1].split('.')[0])
                    filepath = os.path.join(crop_dir, filename)
                    self.samples.append((filepath, label))
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse label from {filename}")
                    continue

        print(f"Loaded {len(self.samples)} digit crops from {len(crop_dirs)} directories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load image (already preprocessed to 28x28)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (add channel dim)
        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mnist_baseline.pth',
                        help='Path to baseline MNIST checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/mnist_finetuned.pth',
                        help='Output checkpoint path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare datasets
    print("\n" + "="*60)
    print("PREPARING DATASETS")
    print("="*60)

    # Training: manual labels (hard cases) + pseudo labels train (easy cases)
    train_dirs = [
        'data/manual_labels_train/digit_crops',
        'data/pseudo_labels_train/digit_crops',
    ]

    # Validation: pseudo labels val
    val_dirs = [
        'data/pseudo_labels_val/digit_crops',
    ]

    train_dataset = DigitCropDataset(train_dirs)
    val_dataset = DigitCropDataset(val_dirs)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load baseline model
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL")
    print("="*60)

    model = SimpleMNIST().to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded baseline model from {args.checkpoint}")
    else:
        print(f"Warning: {args.checkpoint} not found, training from scratch")

    # Evaluate baseline on our data
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Baseline validation accuracy: {val_acc*100:.2f}%")

    # Fine-tune
    print("\n" + "="*60)
    print("FINE-TUNING")
    print("="*60)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = val_acc

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"✓ Saved best model (acc={val_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {args.output}")
    print()
    print("Next steps:")
    print("  1. Re-run classification on failed cases with fine-tuned model")
    print("  2. Generate pseudo-labels for remaining failures")
    print("  3. Build multi-head model for final digit prediction")


if __name__ == '__main__':
    main()
