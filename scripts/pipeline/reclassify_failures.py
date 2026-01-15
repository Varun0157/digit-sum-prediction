"""
Re-classify failed cases using fine-tuned MNIST model.
Generate pseudo-labels for cases that now pass sum validation.

Usage:
    python scripts/reclassify_failures.py --model checkpoints/mnist_finetuned.pth --split train
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.baseline import SimpleMNIST


def segment_digits(img, erode_size=0):
    """Segment digits using contour detection."""
    # Threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional erosion
    if erode_size > 0:
        kernel = np.ones((erode_size, erode_size), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes and sort left to right
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:  # Filter noise
            bboxes.append((x, y, w, h))

    bboxes.sort(key=lambda b: b[0])  # Left to right
    return bboxes


def preprocess_digit_mnist_style(digit_crop, target_size=28, erode_kernel=2, padding=4):
    """Preprocess digit crop to match MNIST style."""
    h, w = digit_crop.shape

    # Erode to thin strokes
    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        digit_crop = cv2.erode(digit_crop.astype(np.uint8), kernel, iterations=1)

    # Center using center of mass
    moments = cv2.moments(digit_crop)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = w // 2, h // 2

    # Create centered image with padding
    max_dim = max(h, w) + 2 * padding
    centered = np.zeros((max_dim, max_dim), dtype=np.uint8)

    paste_y = (max_dim - h) // 2
    paste_x = (max_dim - w) // 2
    centered[paste_y:paste_y+h, paste_x:paste_x+w] = digit_crop

    # Resize to 28x28
    resized = cv2.resize(centered, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized


def classify_digit(model, digit_crop, device):
    """Classify a single preprocessed digit."""
    # Normalize
    img = digit_crop.astype(np.float32) / 255.0

    # To tensor
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/mnist_finetuned.pth',
                        help='Fine-tuned MNIST model')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to process')
    parser.add_argument('--output_dir', type=str, default='data/pseudo_labels_round2_{split}',
                        help='Output directory for new pseudo-labels')
    args = parser.parse_args()

    args.output_dir = args.output_dir.format(split=args.split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load fine-tuned model
    print("\n" + "="*60)
    print("LOADING FINE-TUNED MODEL")
    print("="*60)

    model = SimpleMNIST().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {args.model}")

    # Load failed cases (excluding already manually labeled)
    print("\n" + "="*60)
    print(f"LOADING FAILED CASES ({args.split.upper()})")
    print("="*60)

    samples = np.load(f'data/failed_cases_{args.split}/samples.npy')
    labels = np.load(f'data/failed_cases_{args.split}/labels.npy')

    with open(f'data/failed_cases_{args.split}/index_mapping.json') as f:
        mapping = json.load(f)

    print(f"Total failed cases: {len(samples)}")

    # Load manual labels to skip already labeled cases
    manual_labels_file = f'data/manual_labels_{args.split}/labels.json'
    manually_labeled_indices = set()

    if os.path.exists(manual_labels_file):
        with open(manual_labels_file) as f:
            manual_data = json.load(f)
            manually_labeled_indices = {entry['image_idx'] for entry in manual_data}
        print(f"Skipping {len(manually_labeled_indices)} manually labeled cases")

    # Re-classify with fine-tuned model
    print("\n" + "="*60)
    print("RE-CLASSIFYING WITH FINE-TUNED MODEL")
    print("="*60)

    successes = []
    still_failed = []
    skipped = []

    for i in tqdm(range(len(samples)), desc="Processing"):
        # Skip if already manually labeled
        if i in manually_labeled_indices:
            skipped.append(i)
            continue

        img = samples[i]
        label_sum = int(labels[i])

        # Segment
        bboxes = segment_digits(img, erode_size=0)

        if len(bboxes) != 4:
            skipped.append(i)
            continue

        # Classify each digit
        predictions = []
        for (x, y, w, h) in bboxes:
            crop = img[y:y+h, x:x+w]
            processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)
            pred = classify_digit(model, processed, device)
            predictions.append(pred)

        # Sum validation
        pred_sum = sum(predictions)
        if pred_sum == label_sum:
            successes.append({
                'index': i,
                'original_index': mapping['original_indices'][i],
                'digits': predictions,
                'sum': int(label_sum),
            })
        else:
            still_failed.append(i)

    # Save results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total failed cases: {len(samples)}")
    print(f"Skipped (already labeled): {len(manually_labeled_indices)}")
    print(f"Skipped (segmentation): {len(skipped) - len(manually_labeled_indices)}")
    print(f"✓ Now correct: {len(successes)} ({len(successes)/len(samples)*100:.1f}%)")
    print(f"✗ Still failed: {len(still_failed)} ({len(still_failed)/len(samples)*100:.1f}%)")
    print()

    # Save new pseudo-labels
    os.makedirs(f'{args.output_dir}/digit_crops', exist_ok=True)

    print("Saving new pseudo-labels...")
    saved_labels = []

    for entry in tqdm(successes, desc="Saving"):
        i = entry['index']
        img = samples[i]
        bboxes = segment_digits(img, erode_size=0)

        # Save digit crops
        digit_files = []
        for digit_idx, (x, y, w, h) in enumerate(bboxes):
            crop = img[y:y+h, x:x+w]
            processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)

            label = entry['digits'][digit_idx]
            filename = f"img{entry['original_index']:05d}_digit{digit_idx}_label{label}.png"
            filepath = f"{args.output_dir}/digit_crops/{filename}"
            cv2.imwrite(filepath, processed)
            digit_files.append(filename)

        saved_labels.append({
            'image_idx': entry['original_index'],
            'digits': entry['digits'],
            'sum': entry['sum'],
            'digit_files': digit_files,
        })

    # Save metadata
    with open(f'{args.output_dir}/labels.json', 'w') as f:
        json.dump(saved_labels, f, indent=2)

    print(f"\nSaved {len(successes)} new pseudo-labeled images")
    print(f"  Total digit crops: {len(successes) * 4}")
    print(f"  Output: {args.output_dir}/")
    print()
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review improvement in coverage")
    print("2. Optionally label more of the still-failed cases")
    print("3. Combine all labels for multi-head model training")


if __name__ == '__main__':
    main()
