import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def load_data(data_dir: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    samples = np.load(data_dir / "samples.npy")
    labels = np.load(data_dir / "labels.npy")
    return samples, labels


def plot_label_distribution(
    labels: NDArray[np.uint8], output_path: Path, split_name: str
) -> None:
    unique_labels, counts = np.unique(labels, return_counts=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(unique_labels, counts, alpha=0.7, color="steelblue", label="Count")
    ax1.set_xlabel("Sum Value")
    ax1.set_ylabel("Count", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(unique_labels, counts, color="red", marker="o", label="Distribution")
    ax2.set_ylabel("Count (line)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Label Distribution - {split_name}")
    fig.tight_layout()
    plt.savefig(output_path / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pixel_distribution(
    samples: NDArray[np.uint8], output_path: Path, split_name: str
) -> None:
    pixel_values = samples.flatten()
    hist, bins = np.histogram(pixel_values, bins=256)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(bins[:-1], hist, width=bins[1] - bins[0], alpha=0.7, color="steelblue")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(bins[:-1], hist, color="red", linewidth=1.5)
    ax2.set_ylabel("Frequency (line)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Pixel Value Distribution - {split_name}")
    fig.tight_layout()
    plt.savefig(output_path / "pixel_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_images(
    samples: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    output_path: Path,
    split_name: str,
    n_samples: int = 5,
) -> None:
    n_samples = min(n_samples, len(samples))
    indices = np.random.choice(len(samples), n_samples, replace=False)

    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        ax.imshow(samples[idx], cmap="gray")
        ax.set_title(f"Sum: {labels[idx]}", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Random Samples - {split_name}", fontsize=14)
    fig.tight_layout()
    plt.savefig(output_path / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_statistics(
    samples: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    output_path: Path,
    split_name: str,
) -> None:
    unique_labels = np.unique(labels)
    mean_intensities = []
    std_intensities = []

    for label in unique_labels:
        class_samples = samples[labels == label]
        mean_intensities.append(class_samples.mean())
        std_intensities.append(class_samples.std())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(unique_labels, mean_intensities, marker="o", color="steelblue")
    ax1.set_xlabel("Sum Value")
    ax1.set_ylabel("Mean Pixel Intensity")
    ax1.set_title(f"Mean Pixel Intensity per Class - {split_name}")
    ax1.grid(alpha=0.3)

    ax2.plot(unique_labels, std_intensities, marker="o", color="coral")
    ax2.set_xlabel("Sum Value")
    ax2.set_ylabel("Std Pixel Intensity")
    ax2.set_title(f"Pixel Intensity Std per Class - {split_name}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path / "per_class_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_quality_checks(
    samples: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    output_path: Path,
    split_name: str,
) -> None:
    report_lines = [
        f"Data Quality Report - {split_name}",
        "=" * 50,
        f"Dataset size: {len(samples)}",
        f"Image shape: {samples[0].shape}",
        f"Image dtype: {samples.dtype}",
        f"Label dtype: {labels.dtype}",
        "",
        "Image Statistics:",
        f"  Min pixel value: {samples.min()}",
        f"  Max pixel value: {samples.max()}",
        f"  Mean pixel value: {samples.mean():.2f}",
        f"  Std pixel value: {samples.std():.2f}",
        "",
        "Label Statistics:",
        f"  Min label: {labels.min()}",
        f"  Max label: {labels.max()}",
        f"  Mean label: {labels.mean():.2f}",
        f"  Std label: {labels.std():.2f}",
        f"  Unique labels: {len(np.unique(labels))}",
        "",
        "Quality Checks:",
    ]

    all_same_shape = all(s.shape == samples[0].shape for s in samples)
    report_lines.append(f"  All images same shape: {all_same_shape}")

    has_nan = np.isnan(samples).any() or np.isnan(labels).any()
    report_lines.append(f"  Contains NaN: {has_nan}")

    has_inf = np.isinf(samples).any() or np.isinf(labels).any()
    report_lines.append(f"  Contains Inf: {has_inf}")

    label_counts = np.bincount(labels)
    imbalance_ratio = (
        label_counts.max() / label_counts.min()
        if label_counts.min() > 0
        else float("inf")
    )
    report_lines.append(f"  Class imbalance ratio: {imbalance_ratio:.2f}")

    report = "\n".join(report_lines)
    print(report)

    with open(output_path / "quality_report.txt", "w") as f:
        f.write(report)


def analyse_split(split_dir: Path, output_dir: Path, split_name: str) -> None:
    print(f"\nAnalyzing {split_name}...")

    output_path = output_dir / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    samples, labels = load_data(split_dir)

    plot_label_distribution(labels, output_path, split_name)
    print(f"  Saved label distribution")

    plot_pixel_distribution(samples, output_path, split_name)
    print(f"  Saved pixel distribution")

    plot_sample_images(samples, labels, output_path, split_name)
    print(f"  Saved sample images")

    plot_per_class_statistics(samples, labels, output_path, split_name)
    print(f"  Saved per-class statistics")

    run_quality_checks(samples, labels, output_path, split_name)
    print(f"  Saved quality report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze processed digit sum dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/analysis",
        help="Output directory for analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {data_dir}")
        return

    for subdir in subdirs:
        split_name = subdir.name
        analyse_split(subdir, output_dir, split_name)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
