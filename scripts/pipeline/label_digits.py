"""
Simple tool to manually label digits for fine-tuning MNIST.
Shows image, you type the 4 digits, saves labels.
"""

import sys
sys.path.append('scripts/experiments')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import json
import os
from test_mnist_preprocessing import segment_digits, preprocess_digit_mnist_style

class DigitLabeler:
    def __init__(self, samples, labels, save_path='data/manual_labels', start_idx=0, num_to_label=500):
        self.samples = samples
        self.labels = labels
        self.save_path = save_path
        self.current_idx = start_idx
        self.num_to_label = num_to_label
        self.labeled_data = []

        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/digit_crops', exist_ok=True)

        # Load existing labels if any
        self.label_file = f'{save_path}/labels.json'
        if os.path.exists(self.label_file):
            with open(self.label_file, 'r') as f:
                self.labeled_data = json.load(f)
            print(f"Loaded {len(self.labeled_data)} existing labels")
            self.current_idx = start_idx + len(self.labeled_data)

        self.setup_ui()

    def setup_ui(self):
        """Setup matplotlib UI."""
        self.fig = plt.figure(figsize=(16, 8))

        # Image display (left side)
        self.ax_img = plt.subplot2grid((3, 8), (0, 0), colspan=2, rowspan=3)

        # Digit crops (middle)
        self.ax_digits = [
            plt.subplot2grid((3, 8), (0, 2)),
            plt.subplot2grid((3, 8), (0, 3)),
            plt.subplot2grid((3, 8), (0, 4)),
            plt.subplot2grid((3, 8), (0, 5)),
        ]

        # Instructions (right side)
        self.ax_info = plt.subplot2grid((3, 8), (0, 6), colspan=2, rowspan=3)
        self.ax_info.axis('off')

        # Text input box
        ax_textbox = plt.axes([0.4, 0.02, 0.15, 0.05])
        self.textbox = TextBox(ax_textbox, 'Type 4 digits:', initial='')
        self.textbox.on_submit(self.on_submit)

        # Buttons
        ax_skip = plt.axes([0.58, 0.02, 0.08, 0.05])
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_skip.on_clicked(self.on_skip)

        ax_quit = plt.axes([0.68, 0.02, 0.08, 0.05])
        self.btn_quit = Button(ax_quit, 'Save & Quit')
        self.btn_quit.on_clicked(self.on_quit)

        self.show_current_image()

    def show_current_image(self):
        """Display current image and segmented digits."""
        if self.current_idx >= len(self.samples) or len(self.labeled_data) >= self.num_to_label:
            print(f"\n✓ Labeling complete! Labeled {len(self.labeled_data)} images")
            self.save_labels()
            plt.close()
            return

        # Clear previous
        self.ax_img.clear()
        for ax in self.ax_digits:
            ax.clear()
            ax.axis('off')
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Get current image
        img = self.samples[self.current_idx]
        true_sum = self.labels[self.current_idx]

        # Segment digits
        bboxes = segment_digits(img, erode_size=0)

        # Show original image with bboxes
        img_vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

        self.ax_img.imshow(img_vis)
        self.ax_img.set_title(f'Image #{self.current_idx}', fontsize=12, fontweight='bold')
        self.ax_img.axis('off')

        # Show segmented & preprocessed digits
        self.current_bboxes = bboxes
        self.current_crops = []

        for i, (x, y, w, h) in enumerate(bboxes[:4]):  # Show up to 4 digits
            crop = img[y:y+h, x:x+w]
            processed = preprocess_digit_mnist_style(crop, erode_kernel=2, padding=4)
            self.current_crops.append((crop, processed))

            self.ax_digits[i].imshow(processed, cmap='gray')
            self.ax_digits[i].set_title(f'Digit {i+1}', fontsize=10)

        # Instructions
        progress = len(self.labeled_data)
        total = self.num_to_label
        percent = (progress / total) * 100

        info_text = f"""
LABELING TOOL

Progress: {progress}/{total} ({percent:.1f}%)

┌─────────────────────────┐
│ GROUND TRUTH SUM: {true_sum:2d}   │
└─────────────────────────┘

Instructions:
1. Look at the 4 digits above
2. Type what you see (e.g., "3450")
3. Check: do they sum to {true_sum}?
4. Press ENTER to save
5. Click SKIP if unclear/wrong

Detected: {len(bboxes)} contours

Shortcuts:
  ENTER = Submit
  ESC   = Skip
"""
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.draw()

    def on_submit(self, text):
        """Handle digit submission."""
        text = text.strip()

        # Validate input
        if len(text) != 4 or not text.isdigit():
            print(f"Invalid input: '{text}'. Please enter exactly 4 digits.")
            self.textbox.set_val('')
            return

        digits = [int(d) for d in text]
        predicted_sum = sum(digits)
        true_sum = self.labels[self.current_idx]

        # Warn if sum doesn't match
        if predicted_sum != true_sum:
            print(f"⚠ WARNING: {text} sums to {predicted_sum}, but expected {true_sum}")
            print(f"  Check your digits or click Skip if image is bad")

        # Save label data
        label_entry = {
            'image_idx': int(self.current_idx),
            'digits': digits,
            'true_sum': int(true_sum),
            'predicted_sum': int(predicted_sum),
            'match': bool(predicted_sum == true_sum),
            'num_contours': len(self.current_bboxes),
        }

        # Save individual digit crops
        digit_files = []
        for digit_idx, (crop, processed) in enumerate(self.current_crops[:4]):
            filename = f'img{self.current_idx:05d}_digit{digit_idx}_label{digits[digit_idx]}.png'
            filepath = f'{self.save_path}/digit_crops/{filename}'
            cv2.imwrite(filepath, processed)
            digit_files.append(filename)

        label_entry['digit_files'] = digit_files

        self.labeled_data.append(label_entry)

        # Print feedback
        status = '✓' if predicted_sum == true_sum else '✗'
        print(f"Image {self.current_idx}: {text} = {predicted_sum} (expected={true_sum}) {status}")

        # Save periodically
        if len(self.labeled_data) % 10 == 0:
            self.save_labels()
            print(f"  [Auto-saved: {len(self.labeled_data)} labels]")

        # Next image
        self.current_idx += 1
        self.textbox.set_val('')
        self.show_current_image()

    def on_skip(self, event):
        """Skip current image."""
        print(f"Skipped image {self.current_idx}")
        self.current_idx += 1
        self.textbox.set_val('')
        self.show_current_image()

    def on_quit(self, event):
        """Save and quit."""
        self.save_labels()
        print(f"\n✓ Saved {len(self.labeled_data)} labels to {self.label_file}")
        plt.close()

    def save_labels(self):
        """Save labels to JSON file."""
        with open(self.label_file, 'w') as f:
            json.dump(self.labeled_data, f, indent=2)


def run_labeling_tool(start_idx=0, num_to_label=500):
    """Start the labeling tool."""
    print("="*60)
    print("DIGIT LABELING TOOL")
    print("="*60)
    print(f"Target: Label {num_to_label} images")
    print(f"Starting from image {start_idx}")
    print()
    print("Loading data...")

    # Load dataset
    samples = np.load('data/processed/train/samples.npy')
    labels = np.load('data/processed/train/labels.npy')

    print(f"Loaded {len(samples)} images")
    print()
    print("Starting labeling tool...")
    print("Close the window or click 'Save & Quit' when done.")
    print()

    # Start labeler
    labeler = DigitLabeler(samples, labels, start_idx=start_idx, num_to_label=num_to_label)
    labeler.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Label digits for MNIST fine-tuning')
    parser.add_argument('--start', type=int, default=0, help='Starting image index')
    parser.add_argument('--num', type=int, default=500, help='Number of images to label')
    args = parser.parse_args()

    run_labeling_tool(start_idx=args.start, num_to_label=args.num)
