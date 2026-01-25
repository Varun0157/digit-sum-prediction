"""
Create a combined dataset with ALL available labelled samples for final training.
Combines train, val, and test splits from data/multi into a single training set.

Note: 14 samples in test have digit_labels=[0,0,0,0] (unlabelled at digit level)
but all samples have valid sum_labels (ground truth).

For sum prediction models (SimpleCNN, MultiBranchCNN): use all 30,000 samples
For multi-head digit models: use --exclude_unlabelled to get 29,986 samples
"""

import argparse
import os

import numpy as np


def create_full_dataset(
    output_dir: str = "data/full",
    data_dir: str = "data/multi",
    exclude_unlabelled: bool = False,
) -> None:
    """Combine train, val, and test into a single dataset."""
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    all_sum_labels = []
    all_digit_labels = []

    for split in ["train", "val", "test"]:
        split_dir = f"{data_dir}/{split}"
        try:
            samples = np.load(f"{split_dir}/samples.npy")
            sum_labels = np.load(f"{split_dir}/sum_labels.npy")
            digit_labels = np.load(f"{split_dir}/digit_labels.npy")

            if exclude_unlabelled:
                # Filter out samples where digit labels don't match sum
                computed_sums = digit_labels.sum(axis=1)
                valid_mask = computed_sums == sum_labels
                n_excluded = (~valid_mask).sum()
                samples = samples[valid_mask]
                sum_labels = sum_labels[valid_mask]
                digit_labels = digit_labels[valid_mask]
                print(f"{split}: {len(samples)} samples ({n_excluded} excluded)")
            else:
                print(f"{split}: {len(samples)} samples")

            all_samples.append(samples)
            all_sum_labels.append(sum_labels)
            all_digit_labels.append(digit_labels)
        except FileNotFoundError as e:
            print(f"{split}: not found ({e}), skipping")

    if not all_samples:
        raise ValueError("No data found!")

    # Combine
    combined_samples = np.concatenate(all_samples, axis=0)
    combined_sum_labels = np.concatenate(all_sum_labels, axis=0)
    combined_digit_labels = np.concatenate(all_digit_labels, axis=0)
    print(f"\nCombined: {len(combined_samples)} samples")

    # Save in the format expected by DigitSumDataset (samples.npy, labels.npy)
    np.save(f"{output_dir}/samples.npy", combined_samples)
    np.save(f"{output_dir}/labels.npy", combined_sum_labels)  # For sum prediction
    np.save(f"{output_dir}/sum_labels.npy", combined_sum_labels)
    np.save(f"{output_dir}/digit_labels.npy", combined_digit_labels)

    print(f"\nSaved to {output_dir}/")
    print(f"  samples.npy: {combined_samples.shape}")
    print(f"  labels.npy: {combined_sum_labels.shape} (for sum prediction)")
    print(f"  digit_labels.npy: {combined_digit_labels.shape} (for multi-head)")

    # Print class distribution
    unique, counts = np.unique(combined_sum_labels, return_counts=True)
    print(f"\nClass distribution: {len(unique)} classes (0-{unique.max()})")
    print(f"  Min count: {counts.min()} (class {unique[counts.argmin()]})")
    print(f"  Max count: {counts.max()} (class {unique[counts.argmax()]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create full dataset for final training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/full",
        help="Output directory for combined dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/multi",
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--exclude_unlabelled",
        action="store_true",
        help="Exclude 14 samples without digit-level labels (use for multi-head models)",
    )
    args = parser.parse_args()

    create_full_dataset(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        exclude_unlabelled=args.exclude_unlabelled,
    )
