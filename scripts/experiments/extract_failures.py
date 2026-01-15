"""
Run MNIST classification on entire dataset and extract failed cases for manual labeling.
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

def classify_entire_dataset():
    """Run MNIST on entire dataset and identify failures."""
    print("="*60)
    print("EXTRACTING FAILED CASES FROM FULL DATASET")
    print("="*60)

    # Load data
    print("\nLoading data...")
    samples = np.load('data/processed/train/samples.npy')
    labels = np.load('data/processed/train/labels.npy')
    print(f"Loaded {len(samples)} images")

    # Load MNIST model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNIST().to(device)
    model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
    model.eval()
    print(f"Using device: {device}")

    # Results
    successes = []
    failures = []
    skipped = []  # Images without exactly 4 contours

    print("\nClassifying all images...")
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
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    total_processed = len(successes) + len(failures)
    print(f"Total images: {len(samples)}")
    print(f"Successfully segmented (4 contours): {total_processed}")
    print(f"  ✓ Correct predictions: {len(successes)} ({len(successes)/total_processed*100:.1f}%)")
    print(f"  ✗ Failed predictions: {len(failures)} ({len(failures)/total_processed*100:.1f}%)")
    print(f"Skipped (not 4 contours): {len(skipped)}")
    print()

    # Save results
    os.makedirs('data/classification_results', exist_ok=True)

    with open('data/classification_results/successes.json', 'w') as f:
        json.dump(successes, f, indent=2)

    with open('data/classification_results/failures.json', 'w') as f:
        json.dump(failures, f, indent=2)

    with open('data/classification_results/skipped.json', 'w') as f:
        json.dump(skipped, f, indent=2)

    print(f"Saved results to data/classification_results/")
    print(f"  - successes.json: {len(successes)} images")
    print(f"  - failures.json: {len(failures)} images")
    print(f"  - skipped.json: {len(skipped)} images")
    print()

    # Save pseudo-labeled digit crops for successful cases
    print("Saving pseudo-labeled digit crops for successful cases...")
    os.makedirs('data/pseudo_labels/digit_crops', exist_ok=True)

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
            filename = f'img{idx:05d}_digit{digit_idx}_label{pred_digit}.png'
            filepath = f'data/pseudo_labels/digit_crops/{filename}'
            cv2.imwrite(filepath, processed)
            digit_files.append(filename)

        pseudo_labels.append({
            'image_idx': idx,
            'digits': success['predictions'],
            'true_sum': success['true_sum'],
            'predicted_sum': success['pred_sum'],
            'digit_files': digit_files,
            'source': 'mnist_pseudo_label'
        })

    # Save pseudo-labels metadata
    with open('data/pseudo_labels/labels.json', 'w') as f:
        json.dump(pseudo_labels, f, indent=2)

    print(f"Saved {len(pseudo_labels)} pseudo-labeled digit sets")
    print(f"  Total digit crops: {len(pseudo_labels) * 4}")
    print()

    # Create subset arrays for failed cases
    failed_indices = [f['idx'] for f in failures]
    failed_samples = samples[failed_indices]
    failed_labels = labels[failed_indices]

    os.makedirs('data/failed_cases', exist_ok=True)
    np.save('data/failed_cases/samples.npy', failed_samples)
    np.save('data/failed_cases/labels.npy', failed_labels)

    # Save index mapping
    with open('data/failed_cases/index_mapping.json', 'w') as f:
        json.dump({
            'original_indices': failed_indices,
            'num_failures': len(failed_indices)
        }, f, indent=2)

    print(f"Created subset for failed cases:")
    print(f"  - data/failed_cases/samples.npy: {len(failed_samples)} images")
    print(f"  - data/failed_cases/labels.npy: {len(failed_labels)} labels")
    print()

    print("="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print(f"Manually label the {len(failures)} failed cases to improve MNIST")
    print(f"This is much more efficient than labeling random samples!")
    print()
    print("Next step:")
    print(f"  python scripts/label_failures.py --num {min(500, len(failures))}")

    return successes, failures, skipped


if __name__ == '__main__':
    classify_entire_dataset()
