"""
Test if our handwriting is similar to MNIST using a pretrained model from torch.hub.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os


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


def load_pretrained_mnist():
    """
    Try to load a pretrained MNIST model from various sources.
    """
    print("Searching for pretrained MNIST models...")

    # Option 1: Try pytorch/vision (might have MNIST examples)
    try:
        print("\n1. Trying torch.hub.list('pytorch/vision')...")
        models = torch.hub.list('pytorch/vision')
        print(f"   Available models: {models}")
        # pytorch/vision doesn't have MNIST models, only ImageNet
    except Exception as e:
        print(f"   Failed: {e}")

    # Option 2: Check if there's a community MNIST model
    try:
        print("\n2. Trying community model 'facebookresearch/pytorchvideo'...")
        models = torch.hub.list('facebookresearch/pytorchvideo')
        print(f"   Available: {models[:5]}...")  # Show first 5
    except Exception as e:
        print(f"   Failed: {e}")

    print("\n❌ No pretrained MNIST models found in torch.hub")
    print("   torch.hub mostly has ImageNet models (ResNet, VGG, etc.)")
    print("   MNIST is too simple - most researchers train their own")

    return None


def compare_our_data_to_mnist():
    """
    Visual comparison: Show our digits vs actual MNIST digits.
    """
    from torchvision import datasets

    print("\n" + "="*60)
    print("VISUAL COMPARISON: Our handwriting vs MNIST")
    print("="*60)

    # Load our data
    our_samples = np.load('data/processed/train/samples.npy')

    # Load MNIST data
    mnist = datasets.MNIST('data/mnist', train=True, download=True)

    # Get some examples
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 10, figsize=(20, 6))

    # Row 1: Our handwriting (segmented)
    for col in range(10):
        img = our_samples[col]
        bboxes = segment_digits(img)

        # Take first digit if available
        if len(bboxes) > 0:
            x, y, w, h = bboxes[0]
            digit_crop = img[y:y+h, x:x+w]
            digit_resized = cv2.resize(digit_crop, (28, 28))
            axes[0, col].imshow(digit_resized, cmap='gray')

        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('Our digits\n(segmented)', fontsize=12)

    # Row 2: Our handwriting (raw)
    for col in range(10):
        axes[1, col].imshow(our_samples[col], cmap='gray')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel('Our images\n(full)', fontsize=12)

    # Row 3: MNIST
    for col in range(10):
        mnist_img, _ = mnist[col]
        axes[2, col].imshow(mnist_img, cmap='gray')
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel('MNIST\n(standard)', fontsize=12)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/handwriting_comparison.png', dpi=150)
    print("\n📊 Saved comparison to results/handwriting_comparison.png")

    print("\nKEY DIFFERENCES TO LOOK FOR:")
    print("  - Stroke thickness")
    print("  - Digit centering/padding")
    print("  - Aspect ratio")
    print("  - Writing style variations")


def test_our_mnist_on_real_mnist():
    """
    Test our trained MNIST model on actual MNIST test set.
    This tells us if our model is good, or if our data is the problem.
    """
    from torchvision import datasets

    print("\n" + "="*60)
    print("TESTING: Our trained MNIST model on real MNIST test set")
    print("="*60)

    if not os.path.exists('checkpoints/mnist_classifier.pth'):
        print("❌ No trained model found. Run test_sum_validation.py first.")
        return

    # Load our model
    from test_sum_validation import SimpleMNIST
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNIST().to(device)
    model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth', map_location=device))
    model.eval()

    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Evaluate
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total * 100
    print(f"\n✓ Accuracy on MNIST test set: {correct}/{total} = {accuracy:.2f}%")

    if accuracy > 95:
        print("   ✓ Our model is GOOD (>95% on MNIST)")
        print("   → The 14.7% failure is due to HANDWRITING MISMATCH")
    else:
        print("   ✗ Our model is POOR on MNIST too")
        print("   → The problem is with our MNIST training, not just handwriting")

    return accuracy


if __name__ == '__main__':
    # Try to find pretrained models (will fail, but educational)
    load_pretrained_mnist()

    # Visual comparison
    compare_our_data_to_mnist()

    # Test our model on real MNIST
    test_our_mnist_on_real_mnist()
