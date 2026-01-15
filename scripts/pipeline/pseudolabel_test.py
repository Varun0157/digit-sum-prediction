"""
Pseudo-label test set using trained multi-head model.

Splits test set into:
- labelled/: Successfully predicted (sum matches ground truth)
- unlabelled/: Failed predictions (sum mismatch) - becomes new test set

Only keeps predictions where sum of predicted digits matches ground truth sum.
This ensures we can trust the individual digit predictions.

Usage:
    python scripts/pipeline/pseudolabel_test.py --checkpoint checkpoints/multihead_resnet_best.pth
    python scripts/pipeline/pseudolabel_test.py --checkpoint checkpoints/multihead_resnet_best.pth --confidence_threshold 0.8
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.multihead import MultiHeadResNet


class TestDataset(Dataset):
    """Dataset for test set (sum labels only)."""

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to test data directory (e.g., 'data/multi/test')
        """
        self.samples_raw = np.load(f'{data_dir}/samples.npy')  # Keep original uint8
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')

        # Normalize to [0, 1] for model
        self.samples_normalized = self.samples_raw.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples_raw)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples_normalized[idx]).unsqueeze(0)  # Add channel dim
        sum_label = torch.tensor(self.sum_labels[idx]).long()

        return img, sum_label, idx


def generate_pseudolabels(model, dataloader, device, raw_samples, confidence_threshold=0.0):
    """
    Generate pseudo-labels for test set and split into labelled/unlabelled.

    Args:
        model: Trained multi-head model
        dataloader: Test dataloader
        device: torch device
        raw_samples: Original uint8 samples array
        confidence_threshold: Minimum confidence per digit

    Returns:
        labelled_data: Dict with samples, digit_labels, sum_labels, confidences
        unlabelled_data: Dict with samples, sum_labels
        stats: Dictionary with statistics
    """
    model.eval()

    labelled_indices = []
    labelled_digits = []
    labelled_confidences = []

    unlabelled_indices = []

    stats = {
        'total': 0,
        'sum_match': 0,
        'sum_mismatch': 0,
        'confidence_filtered': 0,
        'final_labelled': 0,
        'final_unlabelled': 0,
    }

    with torch.no_grad():
        for imgs, sum_labels, indices in tqdm(dataloader, desc="Generating pseudo-labels"):
            imgs = imgs.to(device)
            sum_labels = sum_labels.to(device)

            # Forward pass - get 4 head outputs
            outputs = model(imgs)  # List of 4 x [Nx10]

            # Get predicted digits and confidence for each head
            batch_size = imgs.size(0)
            for i in range(batch_size):
                idx = indices[i].item()
                ground_truth_sum = sum_labels[i].item()
                stats['total'] += 1

                # Extract predictions and confidences for this sample
                predicted_digits = []
                confidences = []

                for head_output in outputs:
                    # Get probabilities
                    probs = F.softmax(head_output[i], dim=0)

                    # Get predicted digit and its confidence
                    confidence, predicted = probs.max(0)
                    predicted_digits.append(predicted.item())
                    confidences.append(confidence.item())

                # Calculate predicted sum
                predicted_sum = sum(predicted_digits)

                # Check if sum matches ground truth
                if predicted_sum == ground_truth_sum:
                    stats['sum_match'] += 1

                    # Check minimum confidence threshold
                    min_confidence = min(confidences)

                    if min_confidence >= confidence_threshold:
                        # Successfully labeled
                        labelled_indices.append(idx)
                        labelled_digits.append(predicted_digits)
                        labelled_confidences.append(confidences)
                        stats['final_labelled'] += 1
                    else:
                        # Filtered by confidence - treat as unlabelled
                        unlabelled_indices.append(idx)
                        stats['confidence_filtered'] += 1
                        stats['final_unlabelled'] += 1
                else:
                    # Sum mismatch - unlabelled
                    unlabelled_indices.append(idx)
                    stats['sum_mismatch'] += 1
                    stats['final_unlabelled'] += 1

    # Build labelled dataset
    labelled_data = {
        'samples': raw_samples[labelled_indices],
        'digit_labels': np.array(labelled_digits, dtype=np.uint8),
        'sum_labels': np.array([dataloader.dataset.sum_labels[i] for i in labelled_indices], dtype=np.uint8),
        'confidences': labelled_confidences,
        'indices': labelled_indices,
    }

    # Build unlabelled dataset
    unlabelled_data = {
        'samples': raw_samples[unlabelled_indices],
        'sum_labels': np.array([dataloader.dataset.sum_labels[i] for i in unlabelled_indices], dtype=np.uint8),
        'indices': unlabelled_indices,
    }

    return labelled_data, unlabelled_data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained multi-head model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/multi/test',
                        help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='data/pseudo_labels_multihead',
                        help='Output directory for pseudo-labels')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (must match training)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence per digit (0.0 = no filtering, only sum validation)')
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

    # Generate pseudo-labels
    print("\n" + "="*60)
    print("GENERATING PSEUDO-LABELS")
    print("="*60)
    print(f"Confidence threshold: {args.confidence_threshold}")
    print("(Only keeping predictions where sum matches ground truth)")

    labelled_data, unlabelled_data, stats = generate_pseudolabels(
        model, test_loader, device, test_dataset.samples_raw, args.confidence_threshold
    )

    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total test images: {stats['total']:,}")
    print(f"\nBreakdown:")
    print(f"  Sum matches ground truth: {stats['sum_match']:,} ({stats['sum_match']/stats['total']*100:.2f}%)")
    print(f"  Sum mismatch: {stats['sum_mismatch']:,} ({stats['sum_mismatch']/stats['total']*100:.2f}%)")

    if args.confidence_threshold > 0:
        print(f"  Confidence filtered (<{args.confidence_threshold}): {stats['confidence_filtered']:,} ({stats['confidence_filtered']/stats['total']*100:.2f}%)")

    print(f"\n✓ LABELLED: {stats['final_labelled']:,} ({stats['final_labelled']/stats['total']*100:.2f}%)")
    print(f"✗ UNLABELLED: {stats['final_unlabelled']:,} ({stats['final_unlabelled']/stats['total']*100:.2f}%)")

    # Analyze confidence distribution
    if labelled_data['confidences']:
        min_confs = [min(conf) for conf in labelled_data['confidences']]
        avg_confs = [sum(conf)/len(conf) for conf in labelled_data['confidences']]

        print("\nConfidence Statistics (Labelled):")
        print(f"  Min confidence - Mean: {np.mean(min_confs):.4f}, Median: {np.median(min_confs):.4f}, Std: {np.std(min_confs):.4f}")
        print(f"  Avg confidence - Mean: {np.mean(avg_confs):.4f}, Median: {np.median(avg_confs):.4f}, Std: {np.std(avg_confs):.4f}")

    # Save data
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)

    # Create output directories
    labelled_dir = f'{args.output_dir}/labelled'
    unlabelled_dir = f'{args.output_dir}/unlabelled'
    os.makedirs(labelled_dir, exist_ok=True)
    os.makedirs(unlabelled_dir, exist_ok=True)

    # Save labelled data
    print(f"\nSaving labelled data to {labelled_dir}/")
    np.save(f'{labelled_dir}/samples.npy', labelled_data['samples'])
    np.save(f'{labelled_dir}/digit_labels.npy', labelled_data['digit_labels'])
    np.save(f'{labelled_dir}/sum_labels.npy', labelled_data['sum_labels'])

    labelled_metadata = {
        'num_samples': len(labelled_data['samples']),
        'image_shape': list(labelled_data['samples'].shape[1:]),
        'num_digits': 4,
        'digit_range': [0, 9],
        'sum_range': [0, 36],
        'source': 'pseudo_labels_multihead',
        'checkpoint': args.checkpoint,
        'confidence_threshold': args.confidence_threshold,
    }
    with open(f'{labelled_dir}/metadata.json', 'w') as f:
        json.dump(labelled_metadata, f, indent=2)

    print(f"  samples.npy: {labelled_data['samples'].shape}")
    print(f"  digit_labels.npy: {labelled_data['digit_labels'].shape}")
    print(f"  sum_labels.npy: {labelled_data['sum_labels'].shape}")
    print(f"  metadata.json")

    # Save unlabelled data (new test set)
    print(f"\nSaving unlabelled data to {unlabelled_dir}/")
    np.save(f'{unlabelled_dir}/samples.npy', unlabelled_data['samples'])
    np.save(f'{unlabelled_dir}/sum_labels.npy', unlabelled_data['sum_labels'])

    unlabelled_metadata = {
        'num_samples': len(unlabelled_data['samples']),
        'image_shape': list(unlabelled_data['samples'].shape[1:]),
        'sum_range': [0, 36],
        'note': 'Failed predictions - use as new test set for next iteration',
    }
    with open(f'{unlabelled_dir}/metadata.json', 'w') as f:
        json.dump(unlabelled_metadata, f, indent=2)

    print(f"  samples.npy: {unlabelled_data['samples'].shape}")
    print(f"  sum_labels.npy: {unlabelled_data['sum_labels'].shape}")
    print(f"  metadata.json")

    # Save statistics
    stats_path = f'{args.output_dir}/statistics.json'
    stats_with_config = {
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'confidence_threshold': args.confidence_threshold,
        'statistics': stats,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_with_config, f, indent=2)
    print(f"\nStatistics: {stats_path}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
