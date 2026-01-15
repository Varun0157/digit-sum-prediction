"""
Reorganize data into 80/10/10 (train/val/test) split.

Prerequisites:
- data/multi/train/ with samples.npy, digit_labels.npy, sum_labels.npy
- data/multi/val/ with samples.npy, digit_labels.npy, sum_labels.npy
- data/multi/test/ with samples.npy, sum_labels.npy, optionally digit_labels.npy

Strategy:
1. Load all train + val (guaranteed to have digit labels)
2. Load test and separate into labeled vs unlabeled
3. Combine all labeled data, shuffle, split 80/10
4. Remaining labeled + unlabeled → new test (mixed labeled/unlabeled)
5. Backup existing data before overwriting

Usage:
    python scripts/pipeline/resplit_8_1_1.py
    python scripts/pipeline/resplit_8_1_1.py --no_backup
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np


def backup_data(data_dir):
    backup_dir = f'{data_dir}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f"Backing up {data_dir} to {backup_dir}/")
    shutil.copytree(data_dir, backup_dir)
    print(f"✓ Backup complete")
    return backup_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/multi')
    parser.add_argument('--no_backup', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("="*60)
    print("REORGANIZING DATA: 80/10/10 SPLIT")
    print("="*60)

    if not args.no_backup:
        print("\n" + "="*60)
        print("BACKUP")
        print("="*60)
        backup_dir = backup_data(args.data_dir)

    print("\n" + "="*60)
    print("LOADING EXISTING DATA")
    print("="*60)

    train_samples = np.load(f'{args.data_dir}/train/samples.npy')
    train_digit_labels = np.load(f'{args.data_dir}/train/digit_labels.npy')
    train_sum_labels = np.load(f'{args.data_dir}/train/sum_labels.npy')
    print(f"Train: {len(train_samples):,}")

    val_samples = np.load(f'{args.data_dir}/val/samples.npy')
    val_digit_labels = np.load(f'{args.data_dir}/val/digit_labels.npy')
    val_sum_labels = np.load(f'{args.data_dir}/val/sum_labels.npy')
    print(f"Val: {len(val_samples):,}")

    test_samples = np.load(f'{args.data_dir}/test/samples.npy')
    test_sum_labels = np.load(f'{args.data_dir}/test/sum_labels.npy')
    test_digit_labels_path = f'{args.data_dir}/test/digit_labels.npy'

    if os.path.exists(test_digit_labels_path):
        test_digit_labels = np.load(test_digit_labels_path)
        labeled_mask = test_digit_labels.sum(axis=1) > 0
        test_labeled_indices = np.where(labeled_mask)[0]
        test_unlabeled_indices = np.where(~labeled_mask)[0]
    else:
        test_digit_labels = np.zeros((len(test_samples), 4), dtype=np.uint8)
        test_labeled_indices = np.array([], dtype=int)
        test_unlabeled_indices = np.arange(len(test_samples))

    print(f"Test: {len(test_samples):,} ({len(test_labeled_indices)} labeled, {len(test_unlabeled_indices)} unlabeled)")

    print("\n" + "="*60)
    print("COMBINING ALL LABELED DATA")
    print("="*60)

    all_samples = np.concatenate([
        train_samples,
        val_samples,
        test_samples[test_labeled_indices]
    ])
    all_digit_labels = np.concatenate([
        train_digit_labels,
        val_digit_labels,
        test_digit_labels[test_labeled_indices]
    ])
    all_sum_labels = np.concatenate([
        train_sum_labels,
        val_sum_labels,
        test_sum_labels[test_labeled_indices]
    ])

    print(f"Total labeled: {len(all_samples):,}")

    print("\n" + "="*60)
    print("SPLITTING 80/10/10")
    print("="*60)

    np.random.seed(args.seed)
    indices = np.arange(len(all_samples))
    np.random.shuffle(indices)

    total = 30000
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)

    train_end = train_size
    val_end = train_size + val_size

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_labeled_indices_new = indices[val_end:]

    print(f"Train: {len(train_indices):,} (target: {train_size:,})")
    print(f"Val: {len(val_indices):,} (target: {val_size:,})")
    print(f"Test (labeled): {len(test_labeled_indices_new):,}")
    print(f"Test (unlabeled): {len(test_unlabeled_indices):,}")
    print(f"Test (total): {len(test_labeled_indices_new) + len(test_unlabeled_indices):,}")

    new_train_samples = all_samples[train_indices]
    new_train_digit_labels = all_digit_labels[train_indices]
    new_train_sum_labels = all_sum_labels[train_indices]

    new_val_samples = all_samples[val_indices]
    new_val_digit_labels = all_digit_labels[val_indices]
    new_val_sum_labels = all_sum_labels[val_indices]

    new_test_labeled_samples = all_samples[test_labeled_indices_new]
    new_test_labeled_digit_labels = all_digit_labels[test_labeled_indices_new]
    new_test_labeled_sum_labels = all_sum_labels[test_labeled_indices_new]

    new_test_unlabeled_samples = test_samples[test_unlabeled_indices]
    new_test_unlabeled_sum_labels = test_sum_labels[test_unlabeled_indices]

    new_test_samples = np.concatenate([
        new_test_labeled_samples,
        new_test_unlabeled_samples
    ])
    new_test_digit_labels = np.concatenate([
        new_test_labeled_digit_labels,
        np.zeros((len(test_unlabeled_indices), 4), dtype=np.uint8)
    ])
    new_test_sum_labels = np.concatenate([
        new_test_labeled_sum_labels,
        new_test_unlabeled_sum_labels
    ])

    print("\n" + "="*60)
    print("SAVING NEW SPLITS")
    print("="*60)

    print("\nSaving train...")
    np.save(f'{args.data_dir}/train/samples.npy', new_train_samples)
    np.save(f'{args.data_dir}/train/digit_labels.npy', new_train_digit_labels)
    np.save(f'{args.data_dir}/train/sum_labels.npy', new_train_sum_labels)

    train_metadata = {
        'num_samples': len(new_train_samples),
        'image_shape': list(new_train_samples.shape[1:]),
        'split_ratio': 0.8,
    }
    with open(f'{args.data_dir}/train/metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    print(f"  samples.npy: {new_train_samples.shape}")
    print(f"  digit_labels.npy: {new_train_digit_labels.shape}")
    print(f"  sum_labels.npy: {new_train_sum_labels.shape}")

    print("\nSaving val...")
    np.save(f'{args.data_dir}/val/samples.npy', new_val_samples)
    np.save(f'{args.data_dir}/val/digit_labels.npy', new_val_digit_labels)
    np.save(f'{args.data_dir}/val/sum_labels.npy', new_val_sum_labels)

    val_metadata = {
        'num_samples': len(new_val_samples),
        'image_shape': list(new_val_samples.shape[1:]),
        'split_ratio': 0.1,
    }
    with open(f'{args.data_dir}/val/metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    print(f"  samples.npy: {new_val_samples.shape}")
    print(f"  digit_labels.npy: {new_val_digit_labels.shape}")
    print(f"  sum_labels.npy: {new_val_sum_labels.shape}")

    print("\nSaving test...")
    np.save(f'{args.data_dir}/test/samples.npy', new_test_samples)
    np.save(f'{args.data_dir}/test/digit_labels.npy', new_test_digit_labels)
    np.save(f'{args.data_dir}/test/sum_labels.npy', new_test_sum_labels)

    test_metadata = {
        'num_samples': len(new_test_samples),
        'num_labeled': len(test_labeled_indices_new),
        'num_unlabeled': len(test_unlabeled_indices),
        'image_shape': list(new_test_samples.shape[1:]),
        'split_ratio': 0.1,
        'note': 'Mixed: labeled samples have digit_labels, unlabeled have zeros',
    }
    with open(f'{args.data_dir}/test/metadata.json', 'w') as f:
        json.dump(test_metadata, f, indent=2)

    print(f"  samples.npy: {new_test_samples.shape}")
    print(f"  digit_labels.npy: {new_test_digit_labels.shape}")
    print(f"  sum_labels.npy: {new_test_sum_labels.shape}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOLD:")
    print(f"  Train: {len(train_samples):,}")
    print(f"  Val: {len(val_samples):,}")
    print(f"  Test: {len(test_samples):,}")

    print(f"\nNEW (80/10/10):")
    print(f"  Train: {len(new_train_samples):,}")
    print(f"  Val: {len(new_val_samples):,}")
    print(f"  Test: {len(new_test_samples):,} ({len(test_labeled_indices_new)} labeled, {len(test_unlabeled_indices)} unlabeled)")

    print(f"\nTotal: {len(new_train_samples) + len(new_val_samples) + len(new_test_samples):,}")
    print(f"Total labeled: {len(new_train_samples) + len(new_val_samples) + len(test_labeled_indices_new):,}")

    if not args.no_backup:
        print(f"\n✓ Backup: {backup_dir}/")
    print(f"✓ Data saved to: {args.data_dir}/")


if __name__ == '__main__':
    main()
