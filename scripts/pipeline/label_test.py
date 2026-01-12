"""
Manual labeling tool for test set.

Shows unlabeled test images and allows manual entry of 4 digits.
Validates that entered digits sum to ground truth.
Saves as digit_labels.npy compatible with train/val format.

Usage:
    python scripts/pipeline/label_test.py
    python scripts/pipeline/label_test.py --start 100 --num 50
"""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox


class TestLabeler:
    """Interactive labeling tool for test set."""

    def __init__(self, data_dir='data/multi/test', start_idx=0, num_to_label=None):
        """
        Args:
            data_dir: Test data directory
            start_idx: Starting index in test set
            num_to_label: Number of images to label (None = all remaining)
        """
        self.data_dir = data_dir
        self.start_idx = start_idx

        # Load test data
        print(f"Loading test data from {data_dir}/")
        self.samples = np.load(f'{data_dir}/samples.npy')
        self.sum_labels = np.load(f'{data_dir}/sum_labels.npy')

        # Check if digit labels already exist
        digit_labels_path = f'{data_dir}/digit_labels.npy'
        if os.path.exists(digit_labels_path):
            print(f"WARNING: {digit_labels_path} already exists!")
            response = input("Load existing labels? (y/n): ")
            if response.lower() == 'y':
                self.digit_labels = np.load(digit_labels_path)
                print(f"Loaded existing digit labels: {self.digit_labels.shape}")
            else:
                self.digit_labels = np.zeros((len(self.samples), 4), dtype=np.uint8)
        else:
            self.digit_labels = np.zeros((len(self.samples), 4), dtype=np.uint8)

        # Determine range to label
        if num_to_label is None:
            self.end_idx = len(self.samples)
        else:
            self.end_idx = min(start_idx + num_to_label, len(self.samples))

        self.current_idx = start_idx
        self.num_labeled = 0

        print(f"\nTotal test samples: {len(self.samples)}")
        print(f"Range to label: {start_idx} to {self.end_idx-1} ({self.end_idx-start_idx} images)")
        print()

        self.setup_ui()

    def setup_ui(self):
        """Setup matplotlib UI."""
        self.fig = plt.figure(figsize=(14, 8))

        # Image display (left side - larger)
        self.ax_img = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=3)

        # Info panel (right side)
        self.ax_info = plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=3)
        self.ax_info.axis('off')

        # Text input box
        ax_textbox = plt.axes([0.35, 0.02, 0.20, 0.05])
        self.textbox = TextBox(ax_textbox, 'Type 4 digits:', initial='')
        self.textbox.on_submit(self.on_submit)

        # Buttons
        ax_skip = plt.axes([0.58, 0.02, 0.10, 0.05])
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_skip.on_clicked(self.on_skip)

        ax_save_quit = plt.axes([0.70, 0.02, 0.12, 0.05])
        self.btn_quit = Button(ax_save_quit, 'Save & Quit')
        self.btn_quit.on_clicked(self.on_quit)

        self.show_current_image()

    def show_current_image(self):
        """Display current image."""
        if self.current_idx >= self.end_idx:
            print(f"\n✓ Labeling complete! Labeled {self.num_labeled} images")
            self.save_labels()
            plt.close()
            return

        # Clear previous
        self.ax_img.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Get current image
        img = self.samples[self.current_idx]
        true_sum = self.sum_labels[self.current_idx]

        # Show image
        self.ax_img.imshow(img, cmap='gray')
        self.ax_img.set_title(f'Test Image #{self.current_idx}', fontsize=14, fontweight='bold')
        self.ax_img.axis('off')

        # Calculate progress
        progress = self.num_labeled
        total = self.end_idx - self.start_idx
        percent = (progress / total) * 100 if total > 0 else 0

        # Instructions
        info_text = f"""
MANUAL LABELING TOOL
Test Set ({len(self.samples)} images)

Progress: {progress}/{total} ({percent:.1f}%)
Current Index: {self.current_idx}/{len(self.samples)-1}

┌─────────────────────────────┐
│  GROUND TRUTH SUM: {true_sum:2d}      │
└─────────────────────────────┘

Instructions:
1. Look at the 4 digits in the image
2. Type what you see (e.g., "3450")
3. Verify they sum to {true_sum}
4. Press ENTER to save
5. Click SKIP if unclear

Tips:
• Read left to right
• Take your time on hard cases
• If digits overlap/unclear, skip
• Labels saved every 10 images

Shortcuts:
  ENTER = Submit & Next
  ESC   = Skip
"""
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.draw()

    def on_submit(self, text):
        """Handle digit submission."""
        text = text.strip()

        # Validate input
        if len(text) != 4 or not text.isdigit():
            print(f"❌ Invalid input: '{text}'. Please enter exactly 4 digits.")
            self.textbox.set_val('')
            return

        digits = [int(d) for d in text]
        predicted_sum = sum(digits)
        true_sum = self.sum_labels[self.current_idx]

        # Validate sum
        if predicted_sum != true_sum:
            print(f"⚠️  WARNING: {text} sums to {predicted_sum}, but expected {true_sum}")
            response = input("    Continue anyway? (y/n): ")
            if response.lower() != 'y':
                self.textbox.set_val('')
                return

        # Save digit labels
        self.digit_labels[self.current_idx] = digits
        self.num_labeled += 1

        # Print feedback
        status = '✓' if predicted_sum == true_sum else '⚠️'
        print(f"Image {self.current_idx}: {text} = {predicted_sum} (expected={true_sum}) {status}")

        # Auto-save periodically
        if self.num_labeled % 10 == 0:
            self.save_labels()
            print(f"  [Auto-saved: {self.num_labeled} labels]")

        # Next image
        self.current_idx += 1
        self.textbox.set_val('')
        self.show_current_image()

    def on_skip(self, event):
        """Skip current image."""
        print(f"⏭️  Skipped image {self.current_idx}")
        self.current_idx += 1
        self.textbox.set_val('')
        self.show_current_image()

    def on_quit(self, event):
        """Save and quit."""
        self.save_labels()
        print(f"\n✓ Saved {self.num_labeled} labels")
        plt.close()

    def save_labels(self):
        """Save digit labels to test directory."""
        output_path = f'{self.data_dir}/digit_labels.npy'
        np.save(output_path, self.digit_labels)
        print(f"💾 Saved digit labels to {output_path}")

        # Also save metadata
        metadata_path = f'{self.data_dir}/labeling_metadata.json'
        metadata = {
            'num_samples': len(self.samples),
            'num_labeled': int(self.num_labeled),
            'labeled_indices': [int(i) for i in range(len(self.samples)) if self.digit_labels[i].sum() > 0],
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Manually label test set digits')
    parser.add_argument('--data_dir', type=str, default='data/multi/test',
                        help='Test data directory')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index in test set')
    parser.add_argument('--num', type=int, default=None,
                        help='Number of images to label (default: all remaining)')
    args = parser.parse_args()

    print("="*60)
    print("TEST SET MANUAL LABELING TOOL")
    print("="*60)
    print()

    # Start labeler
    labeler = TestLabeler(
        data_dir=args.data_dir,
        start_idx=args.start,
        num_to_label=args.num
    )

    # Show UI
    plt.show()

    print("\n" + "="*60)
    print("LABELING SESSION COMPLETE")
    print("="*60)
    print(f"Digit labels saved to: {args.data_dir}/digit_labels.npy")
    print()
    print("Next steps:")
    print("  1. Check data integrity:")
    print("     uv run python scripts/pipeline/check_data_integrity.py")
    print("  2. Test set now has digit labels - can move to train/val")
    print("  3. Retrain model on complete dataset")
    print("="*60)


if __name__ == '__main__':
    main()
