"""
Check data integrity across train/val/test splits:
- Verify no duplicate images across splits
- Validate digit_labels sum to sum_labels (where applicable)

Usage:
    python scripts/pipeline/check_data_integrity.py
    python scripts/pipeline/check_data_integrity.py --data_dir data/multi
"""

import argparse
import hashlib
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def hash_image(img):
    """Create SHA256 hash of image array."""
    return hashlib.sha256(img.tobytes()).hexdigest()


def check_duplicates_across_splits(splits_data):
    """Check for duplicate images across different splits."""
    print("\n" + "="*60)
    print("CHECKING FOR DUPLICATE IMAGES")
    print("="*60)

    # Build hash -> (split, index) mapping
    hash_to_location = defaultdict(list)

    for split_name, data in splits_data.items():
        samples = data['samples']
        print(f"\nHashing {split_name} images...")
        for idx in tqdm(range(len(samples)), desc=f"Hashing {split_name}"):
            img_hash = hash_image(samples[idx])
            hash_to_location[img_hash].append((split_name, idx))

    # Find duplicates
    duplicates = {h: locs for h, locs in hash_to_location.items() if len(locs) > 1}

    if duplicates:
        print(f"\n✗ FOUND {len(duplicates)} DUPLICATE IMAGES!")
        print(f"\nShowing first 10 duplicates:")
        for i, (img_hash, locations) in enumerate(list(duplicates.items())[:10]):
            print(f"\n  Hash: {img_hash[:16]}...")
            for split_name, idx in locations:
                print(f"    - {split_name}[{idx}]")
        return False
    else:
        print(f"\n✓ No duplicate images found across splits")
        return True


def check_digit_sum_consistency(split_name, data):
    """Verify that digit_labels sum to sum_labels."""
    if 'digit_labels' not in data:
        return True, []

    print(f"\n  Checking {split_name}: digit_labels sum == sum_labels...")

    digit_labels = data['digit_labels']
    sum_labels = data['sum_labels']

    mismatches = []
    for idx in range(len(digit_labels)):
        computed_sum = digit_labels[idx].sum()
        expected_sum = sum_labels[idx]

        if computed_sum != expected_sum:
            mismatches.append({
                'index': idx,
                'digit_labels': digit_labels[idx].tolist(),
                'computed_sum': int(computed_sum),
                'expected_sum': int(expected_sum),
            })

    if mismatches:
        print(f"    ✗ Found {len(mismatches)} mismatches!")
        return False, mismatches
    else:
        print(f"    ✓ All {len(digit_labels)} samples match")
        return True, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/multi',
                        help='Data directory containing train/val/test splits')
    args = parser.parse_args()

    print("="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    print(f"\nData directory: {args.data_dir}/")

    # Load all splits
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    splits_data = {}

    # Load train
    print("\nLoading train...")
    train_samples = np.load(f'{args.data_dir}/train/samples.npy')
    train_digit_labels = np.load(f'{args.data_dir}/train/digit_labels.npy')
    train_sum_labels = np.load(f'{args.data_dir}/train/sum_labels.npy')
    splits_data['train'] = {
        'samples': train_samples,
        'digit_labels': train_digit_labels,
        'sum_labels': train_sum_labels,
    }
    print(f"  Train: {len(train_samples):,} samples")

    # Load val
    print("\nLoading val...")
    val_samples = np.load(f'{args.data_dir}/val/samples.npy')
    val_digit_labels = np.load(f'{args.data_dir}/val/digit_labels.npy')
    val_sum_labels = np.load(f'{args.data_dir}/val/sum_labels.npy')
    splits_data['val'] = {
        'samples': val_samples,
        'digit_labels': val_digit_labels,
        'sum_labels': val_sum_labels,
    }
    print(f"  Val: {len(val_samples):,} samples")

    # Load test
    print("\nLoading test...")
    test_samples = np.load(f'{args.data_dir}/test/samples.npy')
    test_sum_labels = np.load(f'{args.data_dir}/test/sum_labels.npy')
    splits_data['test'] = {
        'samples': test_samples,
        'sum_labels': test_sum_labels,
    }
    print(f"  Test: {len(test_samples):,} samples")

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"\nTotal: {total_samples:,} samples")

    # Run checks
    all_checks_passed = True

    # Check 1: No duplicates across splits
    no_duplicates = check_duplicates_across_splits(splits_data)
    all_checks_passed = all_checks_passed and no_duplicates

    # Check 2: Digit labels sum to sum labels
    print("\n" + "="*60)
    print("CHECKING DIGIT SUM CONSISTENCY")
    print("="*60)

    all_sum_matches = True
    all_mismatches = {}

    for split_name, data in splits_data.items():
        is_valid, mismatches = check_digit_sum_consistency(split_name, data)
        all_sum_matches = all_sum_matches and is_valid
        if mismatches:
            all_mismatches[split_name] = mismatches

    if all_sum_matches:
        print(f"\n✓ All digit labels sum correctly")
    else:
        print(f"\n✗ Found sum mismatches!")
        for split_name, mismatches in all_mismatches.items():
            print(f"\n  {split_name}: {len(mismatches)} mismatches (showing first 5):")
            for mismatch in mismatches[:5]:
                print(f"    Index {mismatch['index']}: digits={mismatch['digit_labels']}, "
                      f"computed={mismatch['computed_sum']}, expected={mismatch['expected_sum']}")

    all_checks_passed = all_checks_passed and all_sum_matches

    # Final summary
    print("\n" + "="*60)
    print("INTEGRITY CHECK SUMMARY")
    print("="*60)

    checks = [
        ("No duplicate images", no_duplicates),
        ("Digit sums match", all_sum_matches),
    ]

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")

    print()
    if all_checks_passed:
        print("="*60)
        print("✓ ALL INTEGRITY CHECKS PASSED")
        print("="*60)
    else:
        print("="*60)
        print("✗ SOME INTEGRITY CHECKS FAILED")
        print("="*60)
        print("\nPlease review the failures above and fix the data before proceeding.")

    return 0 if all_checks_passed else 1


if __name__ == '__main__':
    exit(main())
