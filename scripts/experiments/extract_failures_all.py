"""
Run MNIST classification on both train and val datasets.
Extract failures and save pseudo-labels for successful cases.
"""

import sys
sys.path.append('scripts')

import numpy as np
import cv2
import torch
import json
import os
from tqdm import tqdm
from test_sum_validation import SimpleMNIST
from test_mnist_preprocessing import segment_digits, preprocess_digit_mnist_style


def process_dataset(split='train'):
    """Process a dataset split (train or val)."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {split.upper()} DATASET")
    print('='*60)

    # Load data
    print(f"\nLoading {split} data...")
    samples = np.load(f'data/processed/{split}/samples.npy')
    labels = np.load(f'data/processed/{split}/labels.npy')
    print(f"Loaded {len(samples)} images")

    # Load MNIST model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNIST().to(device)
    model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
    model.eval()

    # Results
    successes = []
    failures = []
    skipped = []

    print(f"\nClassifying all {split} images...")
    for i in tqdm(range(len(samples)), desc="Processing"):
        img = samples[i]
        label_sum = labels[i]

        # Segment
        bboxes = segment_digits(img, erode_size=0)

        if len(bboxes) != 4:
            skipped.append({
                'idx': i,
                'num_contours': len(bboxes),
                'true_sum': int(label_sum)
            })
            continue

        # Classify with preprocessing
        preds = []
        for (x, y, w, h) in bboxes:
            crop = img[y:y+h, x:x+w]
            processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)

            # Normalize
            normalized = (processed / 255.0 - 0.1307) / 0.3081
            tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                pred = output.argmax(dim=1).item()

            preds.append(pred)

        pred_sum = sum(preds)

        result = {
            'idx': i,
            'predictions': preds,
            'pred_sum': pred_sum,
            'true_sum': int(label_sum),
            'num_contours': len(bboxes)
        }

        if pred_sum == label_sum:
            successes.append(result)
        else:
            failures.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{split.upper()} RESULTS SUMMARY")
    print('='*60)
    total_processed = len(successes) + len(failures)
    print(f"Total images: {len(samples)}")
    print(f"Successfully segmented (4 contours): {total_processed}")
    print(f"  ✓ Correct predictions: {len(successes)} ({len(successes)/total_processed*100:.1f}%)")
    print(f"  ✗ Failed predictions: {len(failures)} ({len(failures)/total_processed*100:.1f}%)")
    print(f"Skipped (not 4 contours): {len(skipped)}")
    print()

    # Save results
    os.makedirs(f'data/classification_results_{split}', exist_ok=True)

    with open(f'data/classification_results_{split}/successes.json', 'w') as f:
        json.dump(successes, f, indent=2)

    with open(f'data/classification_results_{split}/failures.json', 'w') as f:
        json.dump(failures, f, indent=2)

    with open(f'data/classification_results_{split}/skipped.json', 'w') as f:
        json.dump(skipped, f, indent=2)

    print(f"Saved results to data/classification_results_{split}/")

    # Save pseudo-labeled digit crops for successful cases
    print("\nSaving pseudo-labeled digit crops for successful cases...")
    os.makedirs(f'data/pseudo_labels_{split}/digit_crops', exist_ok=True)

    pseudo_labels = []
    for success in tqdm(successes, desc="Saving success crops"):
        idx = success['idx']
        img = samples[idx]
        bboxes = segment_digits(img, erode_size=0)

        digit_files = []
        for digit_idx, (x, y, w, h) in enumerate(bboxes[:4]):
            crop = img[y:y+h, x:x+w]
            processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)

            # Save with pseudo-label
            pred_digit = success['predictions'][digit_idx]
            filename = f'{split}_img{idx:05d}_digit{digit_idx}_label{pred_digit}.png'
            filepath = f'data/pseudo_labels_{split}/digit_crops/{filename}'
            cv2.imwrite(filepath, processed)
            digit_files.append(filename)

        pseudo_labels.append({
            'image_idx': idx,
            'digits': success['predictions'],
            'true_sum': success['true_sum'],
            'predicted_sum': success['pred_sum'],
            'digit_files': digit_files,
            'source': 'mnist_pseudo_label',
            'split': split
        })

    # Save pseudo-labels metadata
    with open(f'data/pseudo_labels_{split}/labels.json', 'w') as f:
        json.dump(pseudo_labels, f, indent=2)

    print(f"Saved {len(pseudo_labels)} pseudo-labeled digit sets")
    print(f"  Total digit crops: {len(pseudo_labels) * 4}")

    # Create subset arrays for failed cases
    failed_indices = [f['idx'] for f in failures]
    failed_samples = samples[failed_indices]
    failed_labels = labels[failed_indices]

    os.makedirs(f'data/failed_cases_{split}', exist_ok=True)
    np.save(f'data/failed_cases_{split}/samples.npy', failed_samples)
    np.save(f'data/failed_cases_{split}/labels.npy', failed_labels)

    # Save index mapping
    with open(f'data/failed_cases_{split}/index_mapping.json', 'w') as f:
        json.dump({
            'original_indices': failed_indices,
            'num_failures': len(failed_indices),
            'split': split
        }, f, indent=2)

    print(f"\nCreated subset for failed cases:")
    print(f"  - data/failed_cases_{split}/samples.npy: {len(failed_samples)} images")
    print(f"  - data/failed_cases_{split}/labels.npy: {len(failed_labels)} labels")

    return successes, failures, skipped


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract failures from train and/or val datasets')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       choices=['train', 'val'],
                       help='Dataset splits to process')
    args = parser.parse_args()

    print("="*60)
    print("EXTRACTING FAILURES FROM ALL DATASETS")
    print("="*60)
    print(f"Processing splits: {', '.join(args.splits)}")

    all_results = {}
    for split in args.splits:
        successes, failures, skipped = process_dataset(split)
        all_results[split] = {
            'successes': len(successes),
            'failures': len(failures),
            'skipped': len(skipped)
        }

    # Print combined summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    for split, results in all_results.items():
        total = results['successes'] + results['failures']
        if total > 0:
            acc = results['successes'] / total * 100
            print(f"{split.upper()}:")
            print(f"  Pseudo-labeled: {results['successes']} ({acc:.1f}%)")
            print(f"  Need manual labeling: {results['failures']}")
            print(f"  Skipped: {results['skipped']}")
            print()

    print("Next steps:")
    print("  1. Label ~500 failures from train set:")
    print("     uv run python scripts/label_failures.py --split train --num 500")
    print("  2. Optionally label failures from val set:")
    print("     uv run python scripts/label_failures.py --split val --num 200")
    print("  3. Combine all labels and fine-tune MNIST")
