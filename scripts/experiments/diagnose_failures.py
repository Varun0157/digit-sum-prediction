"""
Diagnose failure modes: Is it segmentation or classification?
Show preprocessed digits alongside MNIST for comparison.
"""

import sys
sys.path.append('scripts')

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from test_sum_validation import SimpleMNIST
from test_mnist_preprocessing import segment_digits, preprocess_digit_mnist_style

# Load data
samples = np.load('data/processed/train/samples.npy')
labels = np.load('data/processed/train/labels.npy')

# Load MNIST for comparison
mnist = datasets.MNIST('data/mnist', train=True, download=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMNIST().to(device)
model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
model.eval()

print("Finding examples of CORRECT and INCORRECT predictions...")
print("="*60)

correct_examples = []
incorrect_examples = []

for i in range(500):
    img = samples[i]
    label_sum = labels[i]

    # Segment
    bboxes = segment_digits(img, erode_size=0)

    if len(bboxes) != 4:
        continue

    # Classify with preprocessing
    preds = []
    processed_digits = []

    for (x, y, w, h) in bboxes:
        crop = img[y:y+h, x:x+w]
        processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)
        processed_digits.append(processed)

        # Normalize
        normalized = (processed / 255.0 - 0.1307) / 0.3081
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()

        preds.append(pred)

    pred_sum = sum(preds)

    example = {
        'idx': i,
        'image': img,
        'bboxes': bboxes,
        'processed_digits': processed_digits,
        'predictions': preds,
        'pred_sum': pred_sum,
        'true_sum': label_sum,
        'correct': pred_sum == label_sum
    }

    if pred_sum == label_sum and len(correct_examples) < 5:
        correct_examples.append(example)
    elif pred_sum != label_sum and len(incorrect_examples) < 5:
        incorrect_examples.append(example)

    if len(correct_examples) >= 5 and len(incorrect_examples) >= 5:
        break

print(f"Found {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples")

# Create visualization
fig = plt.figure(figsize=(20, 12))

# Layout: 5 rows for each category (correct/incorrect)
# Each row: original image | 4 preprocessed digits | 4 MNIST reference digits

for category_idx, (examples, category_name) in enumerate([
    (correct_examples, "CORRECT Predictions"),
    (incorrect_examples, "INCORRECT Predictions")
]):
    for row_idx, ex in enumerate(examples):
        row = category_idx * 5 + row_idx

        # Column 0: Original image with bboxes
        ax = plt.subplot2grid((10, 9), (row, 0), colspan=1)
        img_vis = cv2.cvtColor(ex['image'].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in ex['bboxes']:
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        ax.imshow(img_vis)
        ax.axis('off')

        if row_idx == 0:
            ax.set_title(f'{category_name}\nOriginal', fontsize=10, fontweight='bold')
        else:
            ax.set_title('Original', fontsize=8)

        status = '✓' if ex['correct'] else '✗'
        ax.text(0.5, -0.15, f"{status} {ex['predictions']}={ex['pred_sum']} (true={ex['true_sum']})",
                transform=ax.transAxes, ha='center', fontsize=8)

        # Columns 1-4: Our preprocessed digits
        for digit_idx in range(4):
            ax = plt.subplot2grid((10, 9), (row, 1 + digit_idx))
            ax.imshow(ex['processed_digits'][digit_idx], cmap='gray')
            ax.axis('off')

            if row_idx == 0 and digit_idx == 1:
                ax.set_title('Our Digits\n(preprocessed)', fontsize=10, fontweight='bold')

            # Show prediction
            ax.text(0.5, -0.1, f'{ex["predictions"][digit_idx]}',
                   transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')

        # Columns 5-8: MNIST reference digits (matching predicted digits)
        for digit_idx in range(4):
            ax = plt.subplot2grid((10, 9), (row, 5 + digit_idx))

            # Find MNIST example of this digit
            pred_digit = ex['predictions'][digit_idx]
            mnist_idx = 0
            for j in range(len(mnist)):
                if mnist[j][1] == pred_digit:
                    mnist_idx = j
                    break

            mnist_img, _ = mnist[mnist_idx]
            ax.imshow(mnist_img, cmap='gray')
            ax.axis('off')

            if row_idx == 0 and digit_idx == 1:
                ax.set_title('MNIST Reference\n(same digit)', fontsize=10, fontweight='bold')

            ax.text(0.5, -0.1, f'{pred_digit}',
                   transform=ax.transAxes, ha='center', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('results/failure_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved diagnostic visualization to results/failure_diagnosis.png")

print("\nQUESTIONS TO ANSWER FROM VISUALIZATION:")
print("1. Do our preprocessed digits look similar to MNIST?")
print("2. Are incorrect predictions due to:")
print("   - Segmentation cutting off parts of digits?")
print("   - Digits genuinely looking different from MNIST?")
print("   - Touching/merged digits causing wrong crops?")
