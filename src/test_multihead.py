"""
Test script for multi-head ResNet model on test or validation set.

Evaluates sum prediction accuracy using predicted digits from 4 heads.
For validation set, also reports per-digit accuracy.

Usage:
    # Base model
    python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best.pth

    # Large model (width_multiplier=1.25)
    python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w125.pth --width_multiplier 1.25

    # Evaluate on validation set
    python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best.pth --split val

    # Large model on test set
    python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w125.pth --width_multiplier 1.25 --split test
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


class EvalDataset(Dataset):
    """Dataset for evaluation (test or val)."""

    def __init__(self, data_dir, has_digit_labels=False):
        """
        Args:
            data_dir: Path to data directory (e.g., 'data/multi/test' or 'data/multi/val')
            has_digit_labels: Whether this split has digit labels (val=True, test=False)
        """
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')
        self.has_digit_labels = has_digit_labels

        if has_digit_labels:
            self.digit_labels = np.load(f'{data_dir}/digit_labels.npy')

        # Normalize to [0, 1]
        self.samples = self.samples.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).unsqueeze(0)  # Add channel dim
        sum_label = torch.tensor(self.sum_labels[idx]).long()

        if self.has_digit_labels:
            digit_label = torch.from_numpy(self.digit_labels[idx]).long()
            return img, digit_label, sum_label
        else:
            return img, sum_label


def evaluate_set(model, dataloader, device, has_digit_labels=False):
    """Evaluate on test or validation set."""
    model.eval()

    sum_correct = 0
    digit_correct = [0] * 4 if has_digit_labels else None
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if has_digit_labels:
                imgs, digit_labels, sum_labels = batch
                digit_labels = digit_labels.to(device)
            else:
                imgs, sum_labels = batch

            imgs = imgs.to(device)
            sum_labels = sum_labels.to(device)

            # Forward pass - get 4 head outputs
            outputs = model(imgs)  # List of 4 x [Nx10]

            # Get predicted digits from each head
            predicted_digits = []
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                predicted_digits.append(predicted)

                # Calculate per-digit accuracy if available
                if has_digit_labels:
                    digit_correct[i] += predicted.eq(digit_labels[:, i]).sum().item()

            # Sum the predicted digits
            predicted_digits = torch.stack(predicted_digits, dim=1)  # Nx4
            predicted_sums = predicted_digits.sum(dim=1)

            # Calculate sum accuracy
            sum_correct += predicted_sums.eq(sum_labels).sum().item()
            total += sum_labels.size(0)

            all_preds.extend(predicted_sums.cpu().numpy())
            all_labels.extend(sum_labels.cpu().numpy())

    sum_acc = sum_correct / total
    mae = mean_absolute_error(all_labels, all_preds)

    results = {
        'sum_accuracy': sum_acc,
        'mae': mae,
        'predictions': all_preds,
        'labels': all_labels,
    }

    if has_digit_labels:
        digit_accs = [correct / total for correct in digit_correct]
        results['digit_accuracies'] = digit_accs
        results['avg_digit_accuracy'] = sum(digit_accs) / 4

    return results


def plot_confusion_matrix(y_true, y_pred, output_path, title='Confusion Matrix - Multi-Head ResNet'):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Sum')
    plt.ylabel('True Sum')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate (test or val)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: data/multi/{split})')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: results/multihead_resnet_{split})')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (must match training)')
    parser.add_argument('--width_multiplier', type=float, default=1.0,
                        help='Model width multiplier (must match training)')
    args = parser.parse_args()

    # Set defaults based on split
    if args.data_dir is None:
        args.data_dir = f'data/multi/{args.split}'
    if args.output_dir is None:
        output_name = 'multihead_resnet'
        if args.width_multiplier != 1.0:
            output_name += f'_w{args.width_multiplier:.2f}'.replace('.', '')
        args.output_dir = f'results/{output_name}_{args.split}'

    has_digit_labels = (args.split == 'val')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("\n" + "="*60)
    print(f"LOADING {args.split.upper()} DATASET")
    print("="*60)

    eval_dataset = EvalDataset(args.data_dir, has_digit_labels=has_digit_labels)
    print(f"{args.split.capitalize()} samples: {len(eval_dataset)}")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    model = MultiHeadResNet(num_digits=4, dropout=args.dropout, width_multiplier=args.width_multiplier).to(device)
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

    results = evaluate_set(model, eval_loader, device, has_digit_labels=has_digit_labels)

    print(f"\n{args.split.capitalize()} Set Results:")
    print(f"  Sum Accuracy: {results['sum_accuracy']*100:.2f}%")
    print(f"  MAE: {results['mae']:.4f}")

    if has_digit_labels:
        print(f"  Avg Digit Accuracy: {results['avg_digit_accuracy']*100:.2f}%")
        print(f"  Per-Digit Accuracies:")
        for i, acc in enumerate(results['digit_accuracies']):
            print(f"    Digit {i+1}: {acc*100:.2f}%")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(eval_dataset),
        'sum_accuracy': float(results['sum_accuracy']),
        'mae': float(results['mae']),
        'parameters': total_params,
    }

    if has_digit_labels:
        metrics['avg_digit_accuracy'] = float(results['avg_digit_accuracy'])
        metrics['digit_accuracies'] = [float(acc) for acc in results['digit_accuracies']]

    metrics_path = f'{args.output_dir}/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save confusion matrix
    cm_path = f'{args.output_dir}/confusion_matrix.png'
    title = f'Confusion Matrix - Multi-Head ResNet ({args.split.capitalize()} Set)'
    plot_confusion_matrix(results['labels'], results['predictions'], cm_path, title=title)

    # Save raw confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    cm_npy_path = f'{args.output_dir}/confusion_matrix.npy'
    np.save(cm_npy_path, cm)
    print(f"Confusion matrix (numpy) saved to {cm_npy_path}")

    # Save predictions
    predictions_path = f'{args.output_dir}/predictions.npz'
    np.savez(predictions_path, predictions=results['predictions'], labels=results['labels'])
    print(f"Predictions saved to {predictions_path}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
