"""
Test contour detection with morphological erosion to separate touching digits.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load samples
samples = np.load('data/processed/train/samples.npy')
labels = np.load('data/processed/train/labels.npy')

print('Comparing: Original vs Erosion')
print('='*60)

# Test different erosion kernel sizes
kernel_sizes = [0, 2, 3]  # 0 = no erosion
results = {}

for k_size in kernel_sizes:
    contour_counts = []

    for i in range(100):  # Test on 100 samples for better statistics
        img = samples[i]

        # Apply erosion if kernel_size > 0
        if k_size > 0:
            kernel = np.ones((k_size, k_size), np.uint8)
            img_processed = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
        else:
            img_processed = img.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes and filter
        bboxes = [cv2.boundingRect(cnt) for cnt in contours]
        bboxes = [b for b in bboxes if b[2] > 5 and b[3] > 10]  # Filter noise

        contour_counts.append(len(bboxes))

    success_rate = contour_counts.count(4) / len(contour_counts) * 100
    results[k_size] = {
        'success_rate': success_rate,
        'distribution': dict(zip(*np.unique(contour_counts, return_counts=True)))
    }

    print(f'Kernel size {k_size}x{k_size}: {success_rate:.1f}% detected exactly 4 contours')
    print(f'  Distribution: {results[k_size]["distribution"]}')

# Visualize comparison on same samples
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

test_indices = [5, 15]  # Samples that had issues (touching digits)

for row, idx in enumerate(test_indices):
    for col, k_size in enumerate(kernel_sizes):
        img = samples[idx]

        # Apply erosion
        if k_size > 0:
            kernel = np.ones((k_size, k_size), np.uint8)
            img_processed = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
        else:
            img_processed = img.astype(np.uint8)

        # Detect contours
        img_color = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(cnt) for cnt in contours]
        bboxes = sorted(bboxes, key=lambda b: b[0])
        bboxes = [b for b in bboxes if b[2] > 5 and b[3] > 10]

        # Draw boxes
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 1)

        ax = axes[row, col]
        ax.imshow(img_color)
        title = f'Sample {idx}: {len(bboxes)} contours'
        if k_size > 0:
            title += f' (erode {k_size}x{k_size})'
        ax.set_title(title)
        ax.axis('off')

plt.tight_layout()
plt.savefig('results/erosion_comparison.png', dpi=150)
print('\nSaved visualization to results/erosion_comparison.png')
