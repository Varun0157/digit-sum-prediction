"""
Reorganize data after pseudo-labeling:
- Add 90% of newly labelled data to train
- Add 10% of newly labelled data to val
- Move unlabelled data to new test set

Backs up existing data/multi/ to data/multi_backup/ before reorganizing.

Usage:
    python scripts/pipeline/reorganize_expanded_data.py
    python scripts/pipeline/reorganize_expanded_data.py --labelled_dir data/pseudo_labels_multihead/labelled --unlabelled_dir data/pseudo_labels_multihead/unlabelled
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
from tqdm import tqdm


def backup_existing_data(data_dir):
    """Backup existing data/multi/ directory."""
    backup_dir = f'{data_dir}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    if os.path.exists(data_dir):
        print(f"Backing up {data_dir} to {backup_dir}/")
        shutil.copytree(data_dir, backup_dir)
        print(f"✓ Backup complete")
    else:
        print(f"No existing {data_dir} found, skipping backup")

    return backup_dir


def load_data(split_dir, has_digit_labels=True):
    """Load data from a directory."""
    samples = np.load(f'{split_dir}/samples.npy')
    sum_labels = np.load(f'{split_dir}/sum_labels.npy')

    data = {
        'samples': samples,
        'sum_labels': sum_labels,
    }

    if has_digit_labels:
        digit_labels = np.load(f'{split_dir}/digit_labels.npy')
        data['digit_labels'] = digit_labels

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_dir', type=str, default='data/multi',
                        help='Current data directory (train/val/test)')
    parser.add_argument('--labelled_dir', type=str, default='data/pseudo_labels_multihead/labelled',
                        help='Newly labelled data directory')
    parser.add_argument('--unlabelled_dir', type=str, default='data/pseudo_labels_multihead/unlabelled',
                        help='Unlabelled data directory (new test set)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of labelled data to add to val (rest goes to train)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Skip backup of existing data')
    args = parser.parse_args()

    print("="*60)
    print("REORGANIZING EXPANDED DATA")
    print("="*60)
    print(f"\nCurrent data: {args.current_dir}/")
    print(f"New labelled: {args.labelled_dir}/")
    print(f"New unlabelled: {args.unlabelled_dir}/")
    print(f"Val split: {args.val_split*100:.0f}% of labelled data")

    # Backup existing data
    if not args.no_backup:
        print("\n" + "="*60)
        print("BACKUP")
        print("="*60)
        backup_dir = backup_existing_data(args.current_dir)

    # Load existing train and val
    print("\n" + "="*60)
    print("LOADING EXISTING DATA")
    print("="*60)

    existing_train = load_data(f'{args.current_dir}/train', has_digit_labels=True)
    print(f"Existing train: {len(existing_train['samples']):,} samples")

    existing_val = load_data(f'{args.current_dir}/val', has_digit_labels=True)
    print(f"Existing val: {len(existing_val['samples']):,} samples")

    # Load newly labelled data
    print("\n" + "="*60)
    print("LOADING NEWLY LABELLED DATA")
    print("="*60)

    labelled = load_data(args.labelled_dir, has_digit_labels=True)
    print(f"Newly labelled: {len(labelled['samples']):,} samples")

    # Split labelled data into train/val
    np.random.seed(42)
    indices = np.arange(len(labelled['samples']))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * args.val_split)
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    print(f"  → Val: {len(val_indices):,} samples ({args.val_split*100:.0f}%)")
    print(f"  → Train: {len(train_indices):,} samples ({(1-args.val_split)*100:.0f}%)")

    # Load unlabelled data (new test set)
    print("\n" + "="*60)
    print("LOADING UNLABELLED DATA (NEW TEST)")
    print("="*60)

    unlabelled = load_data(args.unlabelled_dir, has_digit_labels=False)
    print(f"Unlabelled: {len(unlabelled['samples']):,} samples")

    # Build new train set
    print("\n" + "="*60)
    print("BUILDING NEW TRAIN SET")
    print("="*60)

    new_train_samples = np.concatenate([
        existing_train['samples'],
        labelled['samples'][train_indices]
    ])
    new_train_digit_labels = np.concatenate([
        existing_train['digit_labels'],
        labelled['digit_labels'][train_indices]
    ])
    new_train_sum_labels = np.concatenate([
        existing_train['sum_labels'],
        labelled['sum_labels'][train_indices]
    ])

    print(f"New train: {len(new_train_samples):,} samples")
    print(f"  - Existing train: {len(existing_train['samples']):,}")
    print(f"  - New labelled (90%): {len(train_indices):,}")

    # Build new val set
    print("\n" + "="*60)
    print("BUILDING NEW VAL SET")
    print("="*60)

    new_val_samples = np.concatenate([
        existing_val['samples'],
        labelled['samples'][val_indices]
    ])
    new_val_digit_labels = np.concatenate([
        existing_val['digit_labels'],
        labelled['digit_labels'][val_indices]
    ])
    new_val_sum_labels = np.concatenate([
        existing_val['sum_labels'],
        labelled['sum_labels'][val_indices]
    ])

    print(f"New val: {len(new_val_samples):,} samples")
    print(f"  - Existing val: {len(existing_val['samples']):,}")
    print(f"  - New labelled (10%): {len(val_indices):,}")

    # New test set (unlabelled only)
    print("\n" + "="*60)
    print("BUILDING NEW TEST SET")
    print("="*60)

    new_test_samples = unlabelled['samples']
    new_test_sum_labels = unlabelled['sum_labels']

    print(f"New test: {len(new_test_samples):,} samples")
    print(f"  - Unlabelled data: {len(new_test_samples):,}")

    # Save new data
    print("\n" + "="*60)
    print("SAVING REORGANIZED DATA")
    print("="*60)

    # Save train
    train_dir = f'{args.current_dir}/train'
    os.makedirs(train_dir, exist_ok=True)

    print(f"\nSaving train to {train_dir}/")
    np.save(f'{train_dir}/samples.npy', new_train_samples)
    np.save(f'{train_dir}/digit_labels.npy', new_train_digit_labels)
    np.save(f'{train_dir}/sum_labels.npy', new_train_sum_labels)

    train_metadata = {
        'num_samples': len(new_train_samples),
        'image_shape': list(new_train_samples.shape[1:]),
        'num_digits': 4,
        'digit_range': [0, 9],
        'sum_range': [0, 36],
        'sources': {
            'original_train': len(existing_train['samples']),
            'pseudo_labels_multihead': len(train_indices),
        },
    }
    with open(f'{train_dir}/metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    print(f"  samples.npy: {new_train_samples.shape}")
    print(f"  digit_labels.npy: {new_train_digit_labels.shape}")
    print(f"  sum_labels.npy: {new_train_sum_labels.shape}")

    # Save val
    val_dir = f'{args.current_dir}/val'
    os.makedirs(val_dir, exist_ok=True)

    print(f"\nSaving val to {val_dir}/")
    np.save(f'{val_dir}/samples.npy', new_val_samples)
    np.save(f'{val_dir}/digit_labels.npy', new_val_digit_labels)
    np.save(f'{val_dir}/sum_labels.npy', new_val_sum_labels)

    val_metadata = {
        'num_samples': len(new_val_samples),
        'image_shape': list(new_val_samples.shape[1:]),
        'num_digits': 4,
        'digit_range': [0, 9],
        'sum_range': [0, 36],
        'sources': {
            'original_val': len(existing_val['samples']),
            'pseudo_labels_multihead': len(val_indices),
        },
    }
    with open(f'{val_dir}/metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    print(f"  samples.npy: {new_val_samples.shape}")
    print(f"  digit_labels.npy: {new_val_digit_labels.shape}")
    print(f"  sum_labels.npy: {new_val_sum_labels.shape}")

    # Save test
    test_dir = f'{args.current_dir}/test'
    os.makedirs(test_dir, exist_ok=True)

    print(f"\nSaving test to {test_dir}/")
    np.save(f'{test_dir}/samples.npy', new_test_samples)
    np.save(f'{test_dir}/sum_labels.npy', new_test_sum_labels)

    test_metadata = {
        'num_samples': len(new_test_samples),
        'image_shape': list(new_test_samples.shape[1:]),
        'sum_range': [0, 36],
        'note': 'Unlabelled data from pseudo-labeling - harder cases',
    }
    with open(f'{test_dir}/metadata.json', 'w') as f:
        json.dump(test_metadata, f, indent=2)

    print(f"  samples.npy: {new_test_samples.shape}")
    print(f"  sum_labels.npy: {new_test_sum_labels.shape}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOLD:")
    print(f"  Train: {len(existing_train['samples']):,}")
    print(f"  Val: {len(existing_val['samples']):,}")
    print(f"  Test: (previous test set backed up)")
    print(f"\nNEW:")
    print(f"  Train: {len(new_train_samples):,} (+{len(train_indices):,})")
    print(f"  Val: {len(new_val_samples):,} (+{len(val_indices):,})")
    print(f"  Test: {len(new_test_samples):,}")
    print(f"\nTotal labeled: {len(new_train_samples) + len(new_val_samples):,}")
    print(f"Total unlabeled: {len(new_test_samples):,}")
    print(f"Grand total: {len(new_train_samples) + len(new_val_samples) + len(new_test_samples):,}")

    if not args.no_backup:
        print(f"\n✓ Backup saved to: {backup_dir}/")
    print(f"✓ New data saved to: {args.current_dir}/")


if __name__ == '__main__':
    main()
