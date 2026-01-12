"""
Training script for multi-head ResNet model.

Usage:
    python -m src.train_multihead --epochs 100 --lr 1e-3 --batch_size 128
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from src.model.multihead import MultiHeadResNet


class MultiDigitDataset(Dataset):
    """Dataset for multi-head digit prediction."""

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to data directory (e.g., 'data/multi/train')
        """
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.digit_labels = np.load(f'{data_dir}/digit_labels.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')

        # Normalize to [0, 1]
        self.samples = self.samples.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).unsqueeze(0)  # Add channel dim
        digits = torch.from_numpy(self.digit_labels[idx]).long()  # 4 digit labels
        sum_label = torch.tensor(self.sum_labels[idx]).long()

        return img, digits, sum_label


def train_epoch(model, dataloader, criterions, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    digit_correct = [0] * 4  # Track accuracy per digit position
    digit_total = 0

    pbar = tqdm(dataloader, desc="Training")
    for imgs, digit_labels, _ in pbar:
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)  # Nx4

        optimizer.zero_grad()

        # Forward pass - get 4 head outputs
        outputs = model(imgs)  # List of 4 x [Nx10]

        # Compute loss for each head
        loss = 0
        for i, (output, criterion) in enumerate(zip(outputs, criterions)):
            loss += criterion(output, digit_labels[:, i])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate per-digit accuracy
        for i, output in enumerate(outputs):
            _, predicted = output.max(1)
            digit_correct[i] += predicted.eq(digit_labels[:, i]).sum().item()
        digit_total += digit_labels.size(0)

        # Update progress bar
        avg_acc = sum(digit_correct) / (4 * digit_total) * 100
        pbar.set_postfix({
            'loss': f'{total_loss/digit_total:.4f}',
            'acc': f'{avg_acc:.2f}%'
        })

    avg_loss = total_loss / len(dataloader.dataset)
    digit_accs = [correct / digit_total for correct in digit_correct]
    avg_acc = sum(digit_accs) / 4

    return avg_loss, digit_accs, avg_acc


def evaluate(model, dataloader, criterions, device):
    """Evaluate on validation/test set."""
    model.eval()

    total_loss = 0
    digit_correct = [0] * 4
    digit_total = 0
    sum_correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, digit_labels, sum_labels in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            digit_labels = digit_labels.to(device)
            sum_labels = sum_labels.to(device)

            # Forward pass
            outputs = model(imgs)

            # Compute loss
            loss = 0
            for i, (output, criterion) in enumerate(zip(outputs, criterions)):
                loss += criterion(output, digit_labels[:, i])
            total_loss += loss.item()

            # Calculate per-digit accuracy
            predicted_digits = []
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                digit_correct[i] += predicted.eq(digit_labels[:, i]).sum().item()
                predicted_digits.append(predicted)

            digit_total += digit_labels.size(0)

            # Calculate sum accuracy
            predicted_digits = torch.stack(predicted_digits, dim=1)  # Nx4
            predicted_sums = predicted_digits.sum(dim=1)
            sum_correct += predicted_sums.eq(sum_labels).sum().item()

            all_preds.extend(predicted_sums.cpu().numpy())
            all_labels.extend(sum_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    digit_accs = [correct / digit_total for correct in digit_correct]
    avg_digit_acc = sum(digit_accs) / 4
    sum_acc = sum_correct / digit_total

    return avg_loss, digit_accs, avg_digit_acc, sum_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--data_dir', type=str, default='data/multi', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--wandb_project', type=str, default='digit-sum-prediction', help='W&B project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name='MultiHeadResNet',
            config={
                'model': 'MultiHeadResNet',
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'dropout': args.dropout,
            }
        )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MultiDigitDataset(f'{args.data_dir}/train')
    val_dataset = MultiDigitDataset(f'{args.data_dir}/val')

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

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

    # Create model
    print("\nInitializing model...")
    model = MultiHeadResNet(num_digits=4, dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterions = [nn.CrossEntropyLoss() for _ in range(4)]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_sum_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_digit_accs, train_avg_acc = train_epoch(
            model, train_loader, criterions, optimizer, device
        )

        # Validate
        val_loss, val_digit_accs, val_avg_digit_acc, val_sum_acc, _, _ = evaluate(
            model, val_loader, criterions, device
        )

        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Avg Digit Acc: {train_avg_acc*100:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, Avg Digit Acc: {val_avg_digit_acc*100:.2f}%, Sum Acc: {val_sum_acc*100:.2f}%")
        print(f"Val Digit Accs: " + " | ".join([f"D{i+1}: {acc*100:.1f}%" for i, acc in enumerate(val_digit_accs)]))

        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/avg_digit_acc': train_avg_acc,
                'val/loss': val_loss,
                'val/avg_digit_acc': val_avg_digit_acc,
                'val/sum_acc': val_sum_acc,
                **{f'val/digit{i+1}_acc': acc for i, acc in enumerate(val_digit_accs)}
            })

        # Save best model
        if val_sum_acc > best_sum_acc:
            best_sum_acc = val_sum_acc
            checkpoint_path = f'{args.checkpoint_dir}/multihead_resnet_best.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (sum_acc={val_sum_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation sum accuracy: {best_sum_acc*100:.2f}%")
    print(f"Model saved to: {args.checkpoint_dir}/multihead_resnet_best.pth")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
