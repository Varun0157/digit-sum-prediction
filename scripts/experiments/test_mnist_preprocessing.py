"""
Test if preprocessing (erosion + centering + padding) improves MNIST classification.
"""

import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from test_sum_validation import SimpleMNIST


def segment_digits(img, erode_size=0):
    """Segment image into digit bounding boxes."""
    if erode_size > 0:
        kernel = np.ones((erode_size, erode_size), np.uint8)
        img = cv2.erode(img.astype(np.uint8), kernel)
    else:
        img = img.astype(np.uint8)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    bboxes = [b for b in bboxes if b[2] > 5 and b[3] > 10]
    return sorted(bboxes, key=lambda b: b[0])


def preprocess_digit_mnist_style(digit_crop, target_size=28, erode_kernel=2, padding=4):
    """
    Preprocess digit crop to match MNIST style:
    1. Erode to thin strokes
    2. Center the digit (center of mass)
    3. Add padding
    4. Resize to 28x28
    """
    # Step 1: Erode to thin strokes
    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        digit_crop = cv2.erode(digit_crop.astype(np.uint8), kernel, iterations=1)

    # Step 2: Compute center of mass
    moments = cv2.moments(digit_crop)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = digit_crop.shape[1] // 2, digit_crop.shape[0] // 2

    # Step 3: Create centered image with padding
    h, w = digit_crop.shape

    # Determine size to fit digit + padding
    max_dim = max(h, w) + 2 * padding

    # Create blank canvas
    centered = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # Calculate paste position to center the digit
    paste_y = (max_dim - h) // 2
    paste_x = (max_dim - w) // 2

    # Paste digit
    centered[paste_y:paste_y+h, paste_x:paste_x+w] = digit_crop

    # Step 4: Resize to target size (28x28)
    resized = cv2.resize(centered, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized


def classify_digit(img_crop, model, device, preprocess_fn=None):
    """Classify a digit crop."""
    if preprocess_fn:
        img_processed = preprocess_fn(img_crop)
    else:
        img_processed = cv2.resize(img_crop, (28, 28))

    # Normalize (MNIST stats)
    img_normalized = (img_processed / 255.0 - 0.1307) / 0.3081

    # To tensor
    img_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        return output.argmax(dim=1).item()


def test_preprocessing_variants():
    """Test different preprocessing configurations."""
    print("Testing Preprocessing Variants")
    print("="*60)

    # Load data
    samples = np.load('data/processed/train/samples.npy')
    labels = np.load('data/processed/train/labels.npy')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNIST().to(device)
    model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
    model.eval()

    # Test configurations
    configs = [
        {'name': 'Baseline (resize only)', 'erode': 0, 'padding': 0},
        {'name': 'Erode only (k=1)', 'erode': 1, 'padding': 0},
        {'name': 'Erode (k=2)', 'erode': 2, 'padding': 0},
        {'name': 'Erode (k=2) + Padding=4', 'erode': 2, 'padding': 4},
        {'name': 'Erode (k=2) + Padding=6', 'erode': 2, 'padding': 6},
        {'name': 'Erode (k=1) + Padding=4', 'erode': 1, 'padding': 4},
    ]

    results = {}

    for config in configs:
        correct = 0
        total = 0

        for i in range(100):
            img = samples[i]
            label_sum = labels[i]

            # Segment
            bboxes = segment_digits(img, erode_size=0)

            if len(bboxes) != 4:
                continue

            total += 1

            # Classify with preprocessing
            if config['erode'] > 0 or config['padding'] > 0:
                preprocess_fn = lambda crop: preprocess_digit_mnist_style(
                    crop, erode_kernel=config['erode'], padding=config['padding']
                )
            else:
                preprocess_fn = None

            preds = []
            for (x, y, w, h) in bboxes:
                crop = img[y:y+h, x:x+w]
                pred = classify_digit(crop, model, device, preprocess_fn)
                preds.append(pred)

            pred_sum = sum(preds)

            if pred_sum == label_sum:
                correct += 1

        if total > 0:
            accuracy = correct / total * 100
            results[config['name']] = {'correct': correct, 'total': total, 'accuracy': accuracy}
            print(f"{config['name']:30s}: {correct:2d}/{total:2d} = {accuracy:5.1f}%")

    # Find best
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best: {best[0]} with {best[1]['accuracy']:.1f}% accuracy")

    return results, best


def visualize_preprocessing():
    """Visualize the effect of preprocessing."""
    samples = np.load('data/processed/train/samples.npy')

    # Get one sample
    img = samples[5]  # Sample that had issues before
    bboxes = segment_digits(img)

    if len(bboxes) < 4:
        print("Sample doesn't have 4 contours, trying another...")
        img = samples[0]
        bboxes = segment_digits(img)

    # Take first digit
    x, y, w, h = bboxes[0]
    digit_crop = img[y:y+h, x:x+w]

    # Apply different preprocessing
    variants = [
        ('Original crop', digit_crop),
        ('Resized only', cv2.resize(digit_crop, (28, 28))),
        ('Erode k=1', preprocess_digit_mnist_style(digit_crop, erode_kernel=1, padding=0)),
        ('Erode k=2', preprocess_digit_mnist_style(digit_crop, erode_kernel=2, padding=0)),
        ('Erode k=2 + Pad=4', preprocess_digit_mnist_style(digit_crop, erode_kernel=2, padding=4)),
        ('Erode k=2 + Pad=6', preprocess_digit_mnist_style(digit_crop, erode_kernel=2, padding=6)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (name, processed) in enumerate(variants):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(processed, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/preprocessing_comparison.png', dpi=150)
    print("\n📊 Saved preprocessing visualization to results/preprocessing_comparison.png")


if __name__ == '__main__':
    # Test variants
    results, best = test_preprocessing_variants()

    # Visualize
    visualize_preprocessing()

    print(f"\n{'='*60}")
    print("CONCLUSION:")
    if best[1]['accuracy'] > 50:
        print(f"✓ Preprocessing HELPS! Best accuracy: {best[1]['accuracy']:.1f}%")
        print(f"  Configuration: {best[0]}")
    else:
        print(f"✗ Still poor performance: {best[1]['accuracy']:.1f}%")
        print("  May need to fine-tune MNIST on our handwriting style")
