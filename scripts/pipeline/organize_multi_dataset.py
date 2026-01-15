"""
Organize all labeled data into data/multi/ for multi-head model training.

Strategy:
- TRAIN: 90% of labeled images (~17,382) with digit labels
- VAL: 10% of labeled images (~1,931) with digit labels
- TEST: ALL unlabeled images (~10,687) - sum-only evaluation

Structure:
data/multi/train/:
- samples.npy: Nx40x168 images
- digit_labels.npy: Nx4 digit labels (0-9 for each digit)
- sum_labels.npy: Nx1 sum labels (0-36)
- metadata.json: Source information

data/multi/val/:
- samples.npy: Nx40x168 images
- digit_labels.npy: Nx4 digit labels (0-9 for each digit)
- sum_labels.npy: Nx1 sum labels (0-36)
- metadata.json: Source information

data/multi/test/:
- samples.npy: Nx40x168 images (all unlabeled)
- sum_labels.npy: Nx1 sum labels (0-36, ground truth for evaluation)
- metadata.json: Source information
"""

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_labels_from_json(json_path):
    """Load labels from JSON metadata file."""
    with open(json_path) as f:
        data = json.load(f)

    labels_dict = {}
    for entry in data:
        img_idx = entry['image_idx']
        digits = entry['digits']
        labels_dict[img_idx] = digits

    return labels_dict


def load_original_data(split):
    """Load original samples and labels from processed data."""
    samples_path = f'data/processed/{split}/samples.npy'
    labels_path = f'data/processed/{split}/labels.npy'
    samples = np.load(samples_path)
    labels = np.load(labels_path)
    return samples, labels


def main():
    print("="*60)
    print("ORGANIZING MULTI-HEAD DATASET")
    print("="*60)
    print("\nStrategy:")
    print("  TRAIN: 90% of labeled images")
    print("  VAL: 10% of labeled images")
    print("  TEST: ALL unlabeled images (sum-only evaluation)")
    print()

    os.makedirs('data/multi/train', exist_ok=True)
    os.makedirs('data/multi/val', exist_ok=True)
    os.makedirs('data/multi/test', exist_ok=True)

    # ========================================================================
    # Step 1: Collect ALL labeled images
    # ========================================================================
    print("="*60)
    print("COLLECTING ALL LABELED IMAGES")
    print("="*60)

    all_labels = {}
    sources = {}

    # Process both original train and val splits
    for orig_split in ['train', 'val']:
        print(f"\nProcessing labels from original {orig_split} split...")

        # Source 1: Round 1 pseudo-labels
        source1_path = f'data/pseudo_labels_{orig_split}/labels.json'
        if os.path.exists(source1_path):
            print(f"  Loading Round 1 pseudo-labels...")
            labels = load_labels_from_json(source1_path)
            for idx, digits in labels.items():
                key = (orig_split, idx)
                all_labels[key] = digits
                sources[key] = f'pseudo_round1_{orig_split}'
            print(f"    Added {len(labels)} labels")

        # Source 2: Manual labels (train only)
        if orig_split == 'train':
            source2_path = f'data/manual_labels_{orig_split}/labels.json'
            if os.path.exists(source2_path):
                print(f"  Loading manual labels...")
                labels = load_labels_from_json(source2_path)
                for idx, digits in labels.items():
                    key = (orig_split, idx)
                    all_labels[key] = digits
                    sources[key] = 'manual'
                print(f"    Added {len(labels)} labels")

        # Source 3: Round 2 recovered
        source3_path = f'data/pseudo_labels_round2_{orig_split}/labels.json'
        if os.path.exists(source3_path):
            print(f"  Loading Round 2 recovered...")
            labels = load_labels_from_json(source3_path)
            for idx, digits in labels.items():
                key = (orig_split, idx)
                all_labels[key] = digits
                sources[key] = f'pseudo_round2_{orig_split}'
            print(f"    Added {len(labels)} labels")

    print(f"\nTotal labeled images: {len(all_labels)}")

    # ========================================================================
    # Step 2: Split labeled data 90/10 for train/val
    # ========================================================================
    print("\n" + "="*60)
    print("SPLITTING LABELED DATA (90/10 TRAIN/VAL)")
    print("="*60)

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    keys = list(all_labels.keys())
    np.random.shuffle(keys)

    # 90/10 split
    split_idx = int(len(keys) * 0.9)
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    print(f"\nTrain keys: {len(train_keys)}")
    print(f"Val keys: {len(val_keys)}")

    # Pre-load samples for each split
    print("\nLoading original samples...")
    samples_cache = {}
    for orig_split in ['train', 'val']:
        samples, labels = load_original_data(orig_split)
        samples_cache[orig_split] = (samples, labels)

    # Build train set
    print("\nBuilding train set...")
    train_samples_list = []
    train_digit_labels_list = []
    train_sum_labels_list = []
    train_source_list = []

    for orig_split, idx in tqdm(train_keys, desc="Processing train"):
        train_samples_list.append(samples_cache[orig_split][0][idx])
        digits = all_labels[(orig_split, idx)]
        train_digit_labels_list.append(digits)
        train_sum_labels_list.append(sum(digits))
        train_source_list.append(sources[(orig_split, idx)])

    train_samples = np.array(train_samples_list, dtype=np.uint8)
    train_digit_labels = np.array(train_digit_labels_list, dtype=np.uint8)
    train_sum_labels = np.array(train_sum_labels_list, dtype=np.uint8)

    # Build val set
    print("Building val set...")
    val_samples_list = []
    val_digit_labels_list = []
    val_sum_labels_list = []
    val_source_list = []

    for orig_split, idx in tqdm(val_keys, desc="Processing val"):
        val_samples_list.append(samples_cache[orig_split][0][idx])
        digits = all_labels[(orig_split, idx)]
        val_digit_labels_list.append(digits)
        val_sum_labels_list.append(sum(digits))
        val_source_list.append(sources[(orig_split, idx)])

    val_samples = np.array(val_samples_list, dtype=np.uint8)
    val_digit_labels = np.array(val_digit_labels_list, dtype=np.uint8)
    val_sum_labels = np.array(val_sum_labels_list, dtype=np.uint8)

    # Save train data
    print("\nSaving train data...")
    np.save('data/multi/train/samples.npy', train_samples)
    np.save('data/multi/train/digit_labels.npy', train_digit_labels)
    np.save('data/multi/train/sum_labels.npy', train_sum_labels)

    train_source_counts = {}
    for s in train_source_list:
        train_source_counts[s] = train_source_counts.get(s, 0) + 1

    train_metadata = {
        'num_samples': len(train_samples),
        'image_shape': list(train_samples.shape[1:]),
        'num_digits': 4,
        'digit_range': [0, 9],
        'sum_range': [0, 36],
        'sources': train_source_counts,
    }

    with open('data/multi/train/metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    print(f"  - samples.npy: {train_samples.shape}")
    print(f"  - digit_labels.npy: {train_digit_labels.shape}")
    print(f"  - sum_labels.npy: {train_sum_labels.shape}")

    # Save val data
    print("\nSaving val data...")
    np.save('data/multi/val/samples.npy', val_samples)
    np.save('data/multi/val/digit_labels.npy', val_digit_labels)
    np.save('data/multi/val/sum_labels.npy', val_sum_labels)

    val_source_counts = {}
    for s in val_source_list:
        val_source_counts[s] = val_source_counts.get(s, 0) + 1

    val_metadata = {
        'num_samples': len(val_samples),
        'image_shape': list(val_samples.shape[1:]),
        'num_digits': 4,
        'digit_range': [0, 9],
        'sum_range': [0, 36],
        'sources': val_source_counts,
    }

    with open('data/multi/val/metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    print(f"  - samples.npy: {val_samples.shape}")
    print(f"  - digit_labels.npy: {val_digit_labels.shape}")
    print(f"  - sum_labels.npy: {val_sum_labels.shape}")

    # ========================================================================
    # Step 3: Collect ALL unlabeled images for test set
    # ========================================================================
    print("\n" + "="*60)
    print("BUILDING TEST SET (ALL UNLABELED)")
    print("="*60)

    # Create set of all labeled indices
    labeled_indices = {
        'train': set(),
        'val': set()
    }
    for orig_split, idx in all_labels.keys():
        labeled_indices[orig_split].add(idx)

    # Collect all unlabeled images
    test_samples_list = []
    test_sum_labels_list = []

    for orig_split in ['train', 'val']:
        print(f"\nProcessing unlabeled from {orig_split}...")
        samples, sum_labels = samples_cache[orig_split]

        unlabeled_count = 0
        for idx in range(len(samples)):
            if idx not in labeled_indices[orig_split]:
                test_samples_list.append(samples[idx])
                test_sum_labels_list.append(sum_labels[idx])
                unlabeled_count += 1

        print(f"  Added {unlabeled_count} unlabeled images")

    test_samples = np.array(test_samples_list, dtype=np.uint8)
    test_sum_labels = np.array(test_sum_labels_list, dtype=np.uint8)

    print(f"\nTotal test samples: {len(test_samples)}")

    # Save test data
    print("\nSaving test data...")
    np.save('data/multi/test/samples.npy', test_samples)
    np.save('data/multi/test/sum_labels.npy', test_sum_labels)

    test_metadata = {
        'num_samples': len(test_samples),
        'image_shape': list(test_samples.shape[1:]),
        'sum_range': [0, 36],
        'note': 'No digit labels - test uses sum prediction only',
    }

    with open('data/multi/test/metadata.json', 'w') as f:
        json.dump(test_metadata, f, indent=2)

    print(f"  - samples.npy: {test_samples.shape}")
    print(f"  - sum_labels.npy: {test_sum_labels.shape}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nTrain: {len(train_samples):,} images (with digit labels)")
    print(f"Val: {len(val_samples):,} images (with digit labels)")
    print(f"Test: {len(test_samples):,} images (sum-only evaluation)")
    print()
    print(f"Total: {len(train_samples) + len(val_samples) + len(test_samples):,} images")
    print()
    print(f"Dataset ready for multi-head model training!")
    print(f"Location: data/multi/")


if __name__ == '__main__':
    main()
