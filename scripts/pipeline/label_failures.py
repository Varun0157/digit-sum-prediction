"""
Label the failed cases identified by extract_failures.py
"""

import sys
sys.path.append('scripts/pipeline')

import numpy as np
import json
from label_digits import DigitLabeler

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Label failed MNIST predictions')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Dataset split to label')
    parser.add_argument('--start', type=int, default=0, help='Starting index in failed cases')
    parser.add_argument('--num', type=int, default=500, help='Number to label')
    args = parser.parse_args()

    print("="*60)
    print(f"LABELING FAILED CASES ({args.split.upper()} SET)")
    print("="*60)

    # Check if failures have been extracted
    try:
        samples = np.load(f'data/failed_cases_{args.split}/samples.npy')
        labels = np.load(f'data/failed_cases_{args.split}/labels.npy')
        with open(f'data/failed_cases_{args.split}/index_mapping.json', 'r') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Failed cases for {args.split} not found!")
        print(f"Please run: python scripts/extract_failures_all.py --splits {args.split}")
        sys.exit(1)

    print(f"Loaded {len(samples)} failed cases from {args.split} set")
    print(f"(These are cases where MNIST sum didn't match)")
    print()
    print(f"Target: Label {min(args.num, len(samples))} images")
    print(f"Starting from index {args.start}")
    print()
    print("Starting labeling tool...")
    print("Note: These are the HARD cases - take your time!")
    print()

    # Start labeler on failed cases
    labeler = DigitLabeler(
        samples,
        labels,
        save_path=f'data/manual_labels_{args.split}',
        start_idx=args.start,
        num_to_label=min(args.num, len(samples))
    )

    # Show the UI
    import matplotlib.pyplot as plt
    plt.show()

    print("\n" + "="*60)
    print("Next steps:")
    print(f"  1. Review labels in data/manual_labels_{args.split}/")
    print("  2. Combine with successful pseudo-labels")
    print("  3. Fine-tune MNIST on this data")
    print("="*60)
