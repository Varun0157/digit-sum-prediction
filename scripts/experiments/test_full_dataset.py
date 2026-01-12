"""
Test optimal preprocessing on larger sample of dataset.
"""

import sys
sys.path.append('scripts')

import numpy as np
import cv2
import torch
from test_sum_validation import SimpleMNIST
from test_mnist_preprocessing import segment_digits, preprocess_digit_mnist_style

# Load data
samples = np.load('data/processed/train/samples.npy')
labels = np.load('data/processed/train/labels.npy')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMNIST().to(device)
model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
model.eval()

print('Testing on 500 samples with optimal preprocessing (Erode k=2 + Pad=4)')
print('='*60)

correct = 0
total = 0

for i in range(500):
    img = samples[i]
    label_sum = labels[i]

    # Segment
    bboxes = segment_digits(img, erode_size=0)

    if len(bboxes) != 4:
        continue

    total += 1

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

    if pred_sum == label_sum:
        correct += 1

print(f'Segmentation: {total}/500 images had exactly 4 contours')
if total > 0:
    acc = correct / total * 100
    print(f'Sum Accuracy: {correct}/{total} = {acc:.1f}%')
    print()
    if acc > 70:
        print('✓ SUCCESS! Preprocessing makes MNIST classifier viable for pseudo-labeling')
        print(f'  With {acc:.1f}% accuracy, we can create reliable digit-level labels')
    else:
        print(f'Accuracy {acc:.1f}% - still room for improvement')
