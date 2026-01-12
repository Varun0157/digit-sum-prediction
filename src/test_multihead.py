"""
Test script for multi-head ResNet model on unlabeled test set.

Evaluates sum prediction accuracy using predicted digits from 4 heads.

Usage:
    python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best.pth
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.multihead import MultiHeadResNet


class TestDataset(Dataset):
    """Dataset for test set (sum labels only)."""

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to test data directory (e.g., 'data/multi/test')
        """
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')

        # Normalize to [0, 1]
        self.samples = self.samples.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).unsqueeze(0)  # Add channel dim
        sum_label = torch.tensor(self.sum_labels[idx]).long()

        return img, sum_label


def evaluate_test_set(model, dataloader, device):
    """Evaluate on test set (sum prediction only)."""
    model.eval()

    sum_correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, sum_labels in tqdm(dataloader, desc="Testing"):
            imgs = imgs.to(device)
            sum_labels = sum_labels.to(device)

            # Forward pass - get 4 head outputs
            outputs = model(imgs)  # List of 4 x [Nx10]

            # Get predicted digits from each head
            predicted_digits = []
            for output in outputs:
                _, predicted = output.max(1)
                predicted_digits.append(predicted)

            # Sum the predicted digits
            predicted_digits = torch.stack(predicted_digits, dim=1)  # Nx4
            predicted_sums = predicted_digits.sum(dim=1)

            # Calculate accuracy
            sum_correct += predicted_sums.eq(sum_labels).sum().item()
            total += sum_labels.size(0)

            all_preds.extend(predicted_sums.cpu().numpy())
            all_labels.extend(sum_labels.cpu().numpy())

    sum_acc = sum_correct / total
    mae = mean_absolute_error(all_labels, all_preds)

    return sum_acc, mae, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Sum')
    plt.ylabel('True Sum')
    plt.title('Confusion Matrix - Multi-Head ResNet (Test Set)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/multi/test',
                        help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/multihead_resnet',
                        help='Output directory for results')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (must match training)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    print("\n" + "="*60)
    print("LOADING TEST DATASET")
    print("="*60)

    test_dataset = TestDataset(args.data_dir)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    model = MultiHeadResNet(num_digits=4, dropout=args.dropout).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Total parameters: {total_params:,}")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    sum_acc, mae, all_preds, all_labels = evaluate_test_set(model, test_loader, device)

    print(f"\nTest Set Results:")
    print(f"  Sum Accuracy: {sum_acc*100:.2f}%")
    print(f"  MAE: {mae:.4f}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'checkpoint': args.checkpoint,
        'test_samples': len(test_dataset),
        'sum_accuracy': float(sum_acc),
        'mae': float(mae),
        'parameters': total_params,
    }

    metrics_path = f'{args.output_dir}/test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save confusion matrix
    cm_path = f'{args.output_dir}/confusion_matrix.png'
    plot_confusion_matrix(all_labels, all_preds, cm_path)

    # Save raw confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_npy_path = f'{args.output_dir}/confusion_matrix.npy'
    np.save(cm_npy_path, cm)
    print(f"Confusion matrix (numpy) saved to {cm_npy_path}")

    # Save predictions
    predictions_path = f'{args.output_dir}/predictions.npz'
    np.savez(predictions_path, predictions=all_preds, labels=all_labels)
    print(f"Predictions saved to {predictions_path}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
