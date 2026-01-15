"""
Test script for spatial attention model.

Usage:
    python -m src.test_spatial_attention --checkpoint checkpoints/spatial_attention_best.pth
    python -m src.test_spatial_attention --checkpoint checkpoints/spatial_attention_best.pth --split val
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.multihead import MultiHeadSpatialAttention


class EvalDataset(Dataset):
    def __init__(self, data_dir, has_digit_labels=False):
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')
        self.has_digit_labels = has_digit_labels
        if has_digit_labels:
            self.digit_labels = np.load(f'{data_dir}/digit_labels.npy')
        self.samples = self.samples.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).unsqueeze(0)
        sum_label = torch.tensor(self.sum_labels[idx]).long()
        if self.has_digit_labels:
            digit_label = torch.from_numpy(self.digit_labels[idx]).long()
            return img, digit_label, sum_label
        return img, sum_label


def evaluate(model, dataloader, device, has_digit_labels=False):
    model.eval()
    sum_correct = 0
    digit_correct = [0] * 4 if has_digit_labels else None
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if has_digit_labels:
                imgs, digit_labels, sum_labels = batch
                digit_labels = digit_labels.to(device)
            else:
                imgs, sum_labels = batch

            imgs = imgs.to(device)
            sum_labels = sum_labels.to(device)

            outputs = model(imgs)

            predicted_digits = []
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                predicted_digits.append(predicted)
                if has_digit_labels:
                    digit_correct[i] += predicted.eq(digit_labels[:, i]).sum().item()

            predicted_digits = torch.stack(predicted_digits, dim=1)
            predicted_sums = predicted_digits.sum(dim=1)

            sum_correct += predicted_sums.eq(sum_labels).sum().item()
            total += sum_labels.size(0)

            all_preds.extend(predicted_sums.cpu().numpy())
            all_labels.extend(sum_labels.cpu().numpy())

    results = {
        'sum_accuracy': sum_correct / total,
        'mae': mean_absolute_error(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels,
    }

    if has_digit_labels:
        digit_accs = [c / total for c in digit_correct]
        results['digit_accuracies'] = digit_accs
        results['avg_digit_accuracy'] = sum(digit_accs) / 4

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f'data/multi/{args.split}'
    if args.output_dir is None:
        args.output_dir = f'results/spatial_attention_{args.split}'

    has_digit_labels = (args.split == 'val')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = EvalDataset(args.data_dir, has_digit_labels=has_digit_labels)
    print(f"Samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = MultiHeadSpatialAttention(num_digits=4, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    results = evaluate(model, loader, device, has_digit_labels)

    print(f"\nSum Accuracy: {results['sum_accuracy']*100:.2f}%")
    print(f"MAE: {results['mae']:.4f}")

    if has_digit_labels:
        print(f"Avg Digit Accuracy: {results['avg_digit_accuracy']*100:.2f}%")
        print("Per-digit accuracies:")
        for i, acc in enumerate(results['digit_accuracies']):
            print(f"  Digit {i+1}: {acc*100:.2f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        'sum_accuracy': results['sum_accuracy'],
        'mae': results['mae'],
        'split': args.split,
        'checkpoint': args.checkpoint,
    }
    if has_digit_labels:
        metrics['avg_digit_accuracy'] = results['avg_digit_accuracy']
        metrics['digit_accuracies'] = results['digit_accuracies']

    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Predicted Sum')
    plt.ylabel('True Sum')
    plt.title(f'Spatial Attention - {args.split.capitalize()} Set')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/confusion_matrix.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
