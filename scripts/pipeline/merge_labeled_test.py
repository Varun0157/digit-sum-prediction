"""
Merge manually labeled test set into train/val splits.

After manually labeling the test set, this script:
1. Loads the labeled test data (with digit_labels.npy)
2. Splits it 90/10 into train/val
3. Merges with existing train/val
4. Clears the test set (or leaves only unlabeled if any remain)

Usage:
    python scripts/pipeline/merge_labeled_test.py
    python scripts/pipeline/merge_labeled_test.py --no_backup
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np


def backup_data(data_dir):
    """Backup existing data/multi/ directory."""
    backup_dir = f'{data_dir}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f"Backing up {data_dir} to {backup_dir}/")
    shutil.copytree(data_dir, backup_dir)
    print(f"✓ Backup complete")
    return backup_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/multi',
                        help='Data directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction to add to val (rest goes to train)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Skip backup')
    args = parser.parse_args()

    print("="*60)
    print("MERGING LABELED TEST SET INTO TRAIN/VAL")
    print("="*60)

    # Backup
    if not args.no_backup:
        print("\n" + "="*60)
        print("BACKUP")
        print("="*60)
        backup_dir = backup_data(args.data_dir)

    # Load test set
    print("\n" + "="*60)
    print("LOADING TEST SET")
    print("="*60)

    test_samples = np.load(f'{args.data_dir}/test/samples.npy')
    test_sum_labels = np.load(f'{args.data_dir}/test/sum_labels.npy')

    # Check if digit labels exist
    digit_labels_path = f'{args.data_dir}/test/digit_labels.npy'
    if not os.path.exists(digit_labels_path):
        print(f"ERROR: {digit_labels_path} not found!")
        print("Please run manual labeling first:")
        print("  uv run python scripts/pipeline/label_test.py")
        return 1

    test_digit_labels = np.load(digit_labels_path)

    print(f"Test samples: {len(test_samples)}")

    # Find labeled indices (non-zero digit labels)
    labeled_mask = test_digit_labels.sum(axis=1) > 0
    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(~labeled_mask)[0]

    print(f"  Labeled: {len(labeled_indices)}")
    print(f"  Unlabeled: {len(unlabeled_indices)}")

    if len(labeled_indices) == 0:
        print("\nERROR: No labeled test samples found!")
        print("Please label test set first:")
        print("  uv run python scripts/pipeline/label_test.py")
        return 1

    # Extract labeled test data
    labeled_samples = test_samples[labeled_indices]
    labeled_digit_labels = test_digit_labels[labeled_indices]
    labeled_sum_labels = test_sum_labels[labeled_indices]

    print(f"\nLabeled test data: {len(labeled_samples)} samples")

    # Split labeled data 90/10
    print("\n" + "="*60)
    print("SPLITTING LABELED TEST DATA")
    print("="*60)

    np.random.seed(42)
    indices = np.arange(len(labeled_samples))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * args.val_split)
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    print(f"To train: {len(train_indices)} ({(1-args.val_split)*100:.0f}%)")
    print(f"To val: {len(val_indices)} ({args.val_split*100:.0f}%)")

    # Load existing train/val
    print("\n" + "="*60)
    print("LOADING EXISTING TRAIN/VAL")
    print("="*60)

    train_samples = np.load(f'{args.data_dir}/train/samples.npy')
    train_digit_labels = np.load(f'{args.data_dir}/train/digit_labels.npy')
    train_sum_labels = np.load(f'{args.data_dir}/train/sum_labels.npy')
    print(f"Existing train: {len(train_samples)}")

    val_samples = np.load(f'{args.data_dir}/val/samples.npy')
    val_digit_labels = np.load(f'{args.data_dir}/val/digit_labels.npy')
    val_sum_labels = np.load(f'{args.data_dir}/val/sum_labels.npy')
    print(f"Existing val: {len(val_samples)}")

    # Merge
    print("\n" + "="*60)
    print("MERGING DATA")
    print("="*60)

    new_train_samples = np.concatenate([train_samples, labeled_samples[train_indices]])
    new_train_digit_labels = np.concatenate([train_digit_labels, labeled_digit_labels[train_indices]])
    new_train_sum_labels = np.concatenate([train_sum_labels, labeled_sum_labels[train_indices]])

    new_val_samples = np.concatenate([val_samples, labeled_samples[val_indices]])
    new_val_digit_labels = np.concatenate([val_digit_labels, labeled_digit_labels[val_indices]])
    new_val_sum_labels = np.concatenate([val_sum_labels, labeled_sum_labels[val_indices]])

    print(f"New train: {len(new_train_samples)} (+{len(train_indices)})")
    print(f"New val: {len(new_val_samples)} (+{len(val_indices)})")

    # Save merged data
    print("\n" + "="*60)
    print("SAVING MERGED DATA")
    print("="*60)

    print("\nSaving train...")
    np.save(f'{args.data_dir}/train/samples.npy', new_train_samples)
    np.save(f'{args.data_dir}/train/digit_labels.npy', new_train_digit_labels)
    np.save(f'{args.data_dir}/train/sum_labels.npy', new_train_sum_labels)

    train_metadata = {
        'num_samples': len(new_train_samples),
        'image_shape': list(new_train_samples.shape[1:]),
        'sources': {
            'previous_train': len(train_samples),
            'labeled_test': len(train_indices),
        }
    }
    with open(f'{args.data_dir}/train/metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    print("\nSaving val...")
    np.save(f'{args.data_dir}/val/samples.npy', new_val_samples)
    np.save(f'{args.data_dir}/val/digit_labels.npy', new_val_digit_labels)
    np.save(f'{args.data_dir}/val/sum_labels.npy', new_val_sum_labels)

    val_metadata = {
        'num_samples': len(new_val_samples),
        'image_shape': list(new_val_samples.shape[1:]),
        'sources': {
            'previous_val': len(val_samples),
            'labeled_test': len(val_indices),
        }
    }
    with open(f'{args.data_dir}/val/metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    # Handle remaining test set
    print("\n" + "="*60)
    print("UPDATING TEST SET")
    print("="*60)

    if len(unlabeled_indices) > 0:
        # Keep unlabeled samples as test
        print(f"Keeping {len(unlabeled_indices)} unlabeled samples as test set")

        new_test_samples = test_samples[unlabeled_indices]
        new_test_sum_labels = test_sum_labels[unlabeled_indices]

        np.save(f'{args.data_dir}/test/samples.npy', new_test_samples)
        np.save(f'{args.data_dir}/test/sum_labels.npy', new_test_sum_labels)

        # Remove digit_labels.npy from test (no longer relevant)
        if os.path.exists(digit_labels_path):
            os.remove(digit_labels_path)
            print("Removed digit_labels.npy from test/")

        test_metadata = {
            'num_samples': len(new_test_samples),
            'image_shape': list(new_test_samples.shape[1:]),
            'note': 'Remaining unlabeled samples',
        }
        with open(f'{args.data_dir}/test/metadata.json', 'w') as f:
            json.dump(test_metadata, f, indent=2)
    else:
        # All test data labeled - create empty test set
        print("All test data labeled! Creating empty test set")

        empty_samples = np.zeros((0,) + test_samples.shape[1:], dtype=test_samples.dtype)
        empty_labels = np.zeros(0, dtype=test_sum_labels.dtype)

        np.save(f'{args.data_dir}/test/samples.npy', empty_samples)
        np.save(f'{args.data_dir}/test/sum_labels.npy', empty_labels)

        if os.path.exists(digit_labels_path):
            os.remove(digit_labels_path)

        test_metadata = {
            'num_samples': 0,
            'note': 'All samples labeled and moved to train/val',
        }
        with open(f'{args.data_dir}/test/metadata.json', 'w') as f:
            json.dump(test_metadata, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBEFORE:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)} ({len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled)")
    print(f"\nAFTER:")
    print(f"  Train: {len(new_train_samples)} (+{len(train_indices)})")
    print(f"  Val: {len(new_val_samples)} (+{len(val_indices)})")
    print(f"  Test: {len(unlabeled_indices)}")
    print(f"\nTotal labeled: {len(new_train_samples) + len(new_val_samples)}")

    if not args.no_backup:
        print(f"\n✓ Backup: {backup_dir}/")
    print(f"✓ Data saved to: {args.data_dir}/")

    return 0


if __name__ == '__main__':
    exit(main())
