"""
Create final 90/10 train/val split from all labelled data for final model training.
Uses stratified split to maintain class distribution.
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split


def create_final_split(
    input_dir: str = "data/full_digit_labelled",
    output_dir: str = "data/final",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create stratified 90/10 train/val split."""
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)

    # Load data
    samples = np.load(f"{input_dir}/samples.npy")
    sum_labels = np.load(f"{input_dir}/labels.npy")
    digit_labels = np.load(f"{input_dir}/digit_labels.npy")

    print(f"Loaded {len(samples)} samples from {input_dir}")

    # Stratified split on sum_labels
    indices = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=sum_labels,
    )

    print(f"Train: {len(train_idx)} samples ({100*(1-val_ratio):.0f}%)")
    print(f"Val: {len(val_idx)} samples ({100*val_ratio:.0f}%)")

    # Save train split
    np.save(f"{output_dir}/train/samples.npy", samples[train_idx])
    np.save(f"{output_dir}/train/labels.npy", sum_labels[train_idx])
    np.save(f"{output_dir}/train/sum_labels.npy", sum_labels[train_idx])
    np.save(f"{output_dir}/train/digit_labels.npy", digit_labels[train_idx])

    # Save val split
    np.save(f"{output_dir}/val/samples.npy", samples[val_idx])
    np.save(f"{output_dir}/val/labels.npy", sum_labels[val_idx])
    np.save(f"{output_dir}/val/sum_labels.npy", sum_labels[val_idx])
    np.save(f"{output_dir}/val/digit_labels.npy", digit_labels[val_idx])

    print(f"\nSaved to {output_dir}/")
    print(f"  train/: {len(train_idx)} samples")
    print(f"  val/: {len(val_idx)} samples")

    # Verify class distribution
    train_unique, train_counts = np.unique(sum_labels[train_idx], return_counts=True)
    val_unique, val_counts = np.unique(sum_labels[val_idx], return_counts=True)
    print(f"\nClass distribution preserved:")
    print(f"  Train: {len(train_unique)} classes")
    print(f"  Val: {len(val_unique)} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create final train/val split")
    parser.add_argument("--input_dir", type=str, default="data/full_digit_labelled")
    parser.add_argument("--output_dir", type=str, default="data/final")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_final_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
