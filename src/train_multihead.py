"""
Training script for multi-head ResNet model.

Usage:
    # Base model (~1.2M params)
    python -m src.train_multihead --epochs 100 --lr 1e-3 --batch_size 128

    # Large model (~2M params)
    python -m src.train_multihead --epochs 100 --lr 1e-3 --batch_size 128 --width_multiplier 1.25

    # With augmentation
    python -m src.train_multihead --epochs 100 --lr 1e-3 --batch_size 128 --augment

    # Kernel size ablation
    python -m src.train_multihead --epochs 100 --kernel_size 3
    python -m src.train_multihead --epochs 100 --kernel_size 5
    python -m src.train_multihead --epochs 100 --kernel_size 7  # default
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from src.model.multihead import MultiHeadResNet


class EarlyStopping:
    """
    Early stopping to stop training when validation metrics stop improving.

    Monitors both validation loss and validation accuracy. Training stops
    if neither metric improves for `patience` consecutive epochs.
    """

    def __init__(self, patience=0):
        """
        Args:
            patience: Number of epochs to wait for improvement (0=disabled)
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.epochs_without_improvement = 0

    def step(self, val_loss, val_acc):
        """
        Check if training should stop.

        Returns:
            (should_stop, improved_loss, improved_acc)
        """
        improved_loss = val_loss < self.best_loss
        improved_acc = val_acc > self.best_acc

        if improved_loss:
            self.best_loss = val_loss
        if improved_acc:
            self.best_acc = val_acc

        if improved_loss or improved_acc:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (self.patience > 0 and
                       self.epochs_without_improvement >= self.patience)

        return should_stop, improved_loss, improved_acc

    def status(self):
        """Return current patience status string."""
        if self.patience > 0:
            return f"{self.epochs_without_improvement}/{self.patience}"
        return None


class GaussianNoise(nn.Module):
    """Add Gaussian noise to image."""
    def __init__(self, std=0.02):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return torch.clamp(x + noise, 0, 1)
        return x


class MultiDigitDataset(Dataset):
    """Dataset for multi-head digit prediction."""

    def __init__(self, data_dir, augment=False):
        """
        Args:
            data_dir: Path to data directory (e.g., 'data/multi/train')
            augment: Apply data augmentation (only for training)
        """
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.digit_labels = np.load(f'{data_dir}/digit_labels.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')

        # Normalize to [0, 1]
        self.samples = self.samples.astype(np.float32) / 255.0

        self.augment = augment
        if augment:
            self.transform = T.Compose([
                T.RandomRotation(degrees=5, fill=0),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    fill=0
                ),
                GaussianNoise(std=0.02),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).unsqueeze(0)  # Add channel dim
        digits = torch.from_numpy(self.digit_labels[idx]).long()  # 4 digit labels
        sum_label = torch.tensor(self.sum_labels[idx]).long()

        if self.augment and self.transform is not None:
            img = self.transform(img)

        return img, digits, sum_label


def train_epoch(model, dataloader, criterions, optimizer, device, sum_loss_weight=0.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    digit_correct = [0] * 4  # Track accuracy per digit position
    digit_total = 0

    pbar = tqdm(dataloader, desc="Training")
    for imgs, digit_labels, sum_labels in pbar:
        imgs = imgs.to(device)
        digit_labels = digit_labels.to(device)  # Nx4
        sum_labels = sum_labels.to(device)

        optimizer.zero_grad()

        # Forward pass - get 4 head outputs
        outputs = model(imgs)  # List of 4 x [Nx10]

        # Compute loss for each head
        digit_loss = 0
        for i, (output, criterion) in enumerate(zip(outputs, criterions)):
            digit_loss += criterion(output, digit_labels[:, i])

        # Compute differentiable sum loss if weight > 0
        if sum_loss_weight > 0:
            digit_values = torch.arange(10, device=device).float()
            expected_digits = []
            for output in outputs:
                probs = torch.softmax(output, dim=1)  # Nx10
                expected = (probs * digit_values).sum(dim=1)  # N
                expected_digits.append(expected)
            predicted_sum = torch.stack(expected_digits, dim=1).sum(dim=1)
            sum_loss = nn.functional.mse_loss(predicted_sum, sum_labels.float())
            loss = digit_loss + sum_loss_weight * sum_loss
        else:
            loss = digit_loss

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
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--width_multiplier', type=float, default=1.0, help='Model width multiplier (1.0=base ~1.2M, 1.25=large ~2M)')
    parser.add_argument('--kernel_size', type=int, default=7, choices=[3, 5, 7], help='Initial conv kernel size (default: 7)')
    parser.add_argument('--sum_loss_weight', type=float, default=0.0, help='Weight for differentiable sum loss (default: 0.0, disabled)')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0=disabled). Stops if neither val loss nor val acc improves for this many epochs.')
    parser.add_argument('--suffix', type=str, default='', help='Custom suffix for checkpoint name (e.g., "full")')
    parser.add_argument('--data_dir', type=str, default='data/multi', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--wandb_project', type=str, default='digit-sum-prediction', help='W&B project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build run name suffix
    name_suffix = []
    if args.kernel_size != 7:
        name_suffix.append(f'k{args.kernel_size}')
    if args.width_multiplier != 1.0:
        name_suffix.append(f'w{args.width_multiplier:.2f}'.replace('.', ''))
    if args.sum_loss_weight > 0:
        name_suffix.append(f'sum{args.sum_loss_weight}'.replace('.', ''))
    if args.augment:
        name_suffix.append('aug')
    if args.suffix:
        name_suffix.append(args.suffix)
    run_name = 'MultiHeadResNet' + ('_' + '_'.join(name_suffix) if name_suffix else '')

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': 'MultiHeadResNet',
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'dropout': args.dropout,
                'augment': args.augment,
                'width_multiplier': args.width_multiplier,
                'kernel_size': args.kernel_size,
                'sum_loss_weight': args.sum_loss_weight,
            }
        )

    # Load datasets
    print("\nLoading datasets...")
    print(f"Data augmentation: {'ENABLED' if args.augment else 'DISABLED'}")
    print(f"Model width multiplier: {args.width_multiplier}")
    print(f"Initial kernel size: {args.kernel_size}")
    print(f"Sum loss weight: {args.sum_loss_weight}")
    train_dataset = MultiDigitDataset(f'{args.data_dir}/train', augment=args.augment)
    val_dataset = MultiDigitDataset(f'{args.data_dir}/val', augment=False)

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
    model = MultiHeadResNet(num_digits=4, dropout=args.dropout, width_multiplier=args.width_multiplier, kernel_size=args.kernel_size).to(device)

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
    early_stopping = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_digit_accs, train_avg_acc = train_epoch(
            model, train_loader, criterions, optimizer, device, args.sum_loss_weight
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

        # Early stopping check
        should_stop, _, improved_acc = early_stopping.step(val_loss, val_sum_acc)

        # Save best model (when accuracy improves)
        if improved_acc:
            checkpoint_name = 'multihead_resnet_best'
            if args.kernel_size != 7:
                checkpoint_name += f'_k{args.kernel_size}'
            if args.width_multiplier != 1.0:
                checkpoint_name += f'_w{args.width_multiplier:.2f}'.replace('.', '')
            if args.sum_loss_weight > 0:
                checkpoint_name += f'_sum{args.sum_loss_weight}'.replace('.', '')
            if args.augment:
                checkpoint_name += '_aug'
            if args.suffix:
                checkpoint_name += f'_{args.suffix}'
            checkpoint_name += '.pth'
            checkpoint_path = f'{args.checkpoint_dir}/{checkpoint_name}'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (sum_acc={val_sum_acc*100:.2f}%)")

        # Print patience status and check for early stop
        if early_stopping.status():
            print(f"  Patience: {early_stopping.status()}")
        if should_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation sum accuracy: {early_stopping.best_acc*100:.2f}%")

    # Print final checkpoint name
    checkpoint_name = 'multihead_resnet_best'
    if args.kernel_size != 7:
        checkpoint_name += f'_k{args.kernel_size}'
    if args.width_multiplier != 1.0:
        checkpoint_name += f'_w{args.width_multiplier:.2f}'.replace('.', '')
    if args.sum_loss_weight > 0:
        checkpoint_name += f'_sum{args.sum_loss_weight}'.replace('.', '')
    if args.augment:
        checkpoint_name += '_aug'
    if args.suffix:
        checkpoint_name += f'_{args.suffix}'
    checkpoint_name += '.pth'
    print(f"Model saved to: {args.checkpoint_dir}/{checkpoint_name}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
