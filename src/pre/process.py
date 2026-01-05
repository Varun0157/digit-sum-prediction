import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


def load_raw_data(data_dir: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    data_files = sorted(data_dir.glob("data*.npy"))
    label_files = sorted(data_dir.glob("lab*.npy"))

    assert len(data_files) > 0, f"No data files found in {data_dir}"
    assert len(data_files) == len(label_files), (
        f"Mismatch: {len(data_files)} data files, {len(label_files)} label files"
    )

    all_samples = [np.load(f) for f in data_files]
    all_labels = [np.load(f) for f in label_files]

    samples = np.concatenate(all_samples, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Loaded {len(data_files)} files: {samples.shape[0]} total samples")
    return samples, labels


def split_data(
    samples: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    val_ratio: float,
    no_val: bool,
    random_state: int,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
]:
    if no_val:
        print("No validation set - all data in train")
        return samples, labels, None, None

    train_samples, val_samples, train_labels, val_labels = train_test_split(
        samples,
        labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=random_state,
    )

    print(f"Stratified split: {len(train_samples)} train, {len(val_samples)} val (ratio={val_ratio:.2f})")
    return train_samples, train_labels, val_samples, val_labels


def save_data(
    output_dir: Path,
    train_samples: NDArray[np.uint8],
    train_labels: NDArray[np.uint8],
    val_samples: NDArray[np.uint8] | None,
    val_labels: NDArray[np.uint8] | None,
) -> None:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "samples.npy", train_samples)
    np.save(train_dir / "labels.npy", train_labels)
    print(f"Saved train data to {train_dir}")

    if val_samples is not None and val_labels is not None:
        val_dir = output_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)

        np.save(val_dir / "samples.npy", val_samples)
        np.save(val_dir / "labels.npy", val_labels)
        print(f"Saved val data to {val_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess digit sum dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--val_rat",
        type=float,
        default=0.2,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Do not create validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    samples, labels = load_raw_data(data_dir)
    train_samples, train_labels, val_samples, val_labels = split_data(
        samples, labels, args.val_rat, args.no_val, args.seed
    )
    save_data(output_dir, train_samples, train_labels, val_samples, val_labels)

    print("Processing complete!")


if __name__ == "__main__":
    main()
