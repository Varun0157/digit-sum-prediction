import argparse

from src.data import get_dataloader
from src.model import SimpleCNN
from src.train import run_model


def get_default_config() -> dict:
    return {
        "batch_size": 32,
        "num_epochs": 200,
        "lr": 1e-3,
        "kernel_size": 5,
        "model": "SimpleCNN",
    }


def defaults(balance: bool = True) -> None:
    config = get_default_config()

    train_loader, train_weights = get_dataloader(
        "data/processed/train",
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader, _ = get_dataloader(
        "data/processed/val",
        batch_size=config["batch_size"],
        shuffle=False,
    )

    num_classes = len(train_weights)
    model = SimpleCNN(num_classes=num_classes, kernel_size=config["kernel_size"])
    weight_mode = "balanced" if balance else "unweighted"
    model_name = f"SimpleCNN_k{config['kernel_size']}_defaults_{weight_mode}"

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, val_loader, weights)


def sanity(balance: bool = True) -> None:
    config = get_default_config()
    config["num_epochs"] = 200

    train_loader, train_weights = get_dataloader(
        "data/processed/train",
        batch_size=config["batch_size"],
        shuffle=True,
    )

    num_classes = len(train_weights)
    model = SimpleCNN(num_classes=num_classes, kernel_size=config["kernel_size"])
    weight_mode = "balanced" if balance else "unweighted"
    model_name = f"SimpleCNN_sanity_check_{weight_mode}"

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, train_loader, weights)


def kernel(balance: bool = True) -> None:
    base_config = get_default_config()
    kernel_sizes = [3, 5, 7]
    weight_mode = "balanced" if balance else "unweighted"

    for kernel_size in kernel_sizes:
        config = {**base_config, "kernel_size": kernel_size}

        train_loader, train_weights = get_dataloader(
            "data/processed/train",
            batch_size=config["batch_size"],
            shuffle=True,
        )
        val_loader, _ = get_dataloader(
            "data/processed/val",
            batch_size=config["batch_size"],
            shuffle=False,
        )

        num_classes = len(train_weights)
        model = SimpleCNN(num_classes=num_classes, kernel_size=kernel_size)
        model_name = f"SimpleCNN_k{kernel_size}_{weight_mode}"

        print(f"\n{'=' * 60}")
        print(f"Training with kernel_size={kernel_size}")
        print(f"{'=' * 60}\n")

        weights = train_weights if balance else None
        run_model(model, model_name, config, train_loader, val_loader, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="sanity",
        choices=["sanity", "defaults", "kernel"],
        help="Training mode to run",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Use class weights for balancing",
    )
    args = parser.parse_args()

    match args.mode:
        case "sanity":
            sanity(balance=args.balance)
        case "defaults":
            defaults(balance=args.balance)
        case "kernel":
            kernel(balance=args.balance)
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")
