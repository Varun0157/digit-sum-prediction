import argparse

from src.data import get_dataloader
from src.model import SimpleCNN
from src.train import run_model


def get_default_config() -> dict:
    return {
        "batch_size": 128,
        "num_epochs": 200,
        "lr": 1e-3,
        "kernel_size": 5,
        "pool_type": "max",
        "model": "SimpleCNN",
    }


def get_name(config: dict, balance: bool, suffix: str = "") -> str:
    weight_mode = "balanced" if balance else "unweighted"
    pool_type = config.get("pool_type", "max")
    name = f"SimpleCNN_k{config['kernel_size']}_{pool_type}_{weight_mode}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def defaults(balance: bool = True, pool_type: str = "max") -> None:
    config = get_default_config()
    config["pool_type"] = pool_type

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
    model = SimpleCNN(
        num_classes=num_classes,
        kernel_size=config["kernel_size"],
        pool_type=config["pool_type"],
    )
    model_name = get_name(config, balance)

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, val_loader, weights)


def sanity(balance: bool = True, pool_type: str = "max") -> None:
    config = get_default_config()
    config["num_epochs"] = 200
    config["pool_type"] = pool_type

    train_loader, train_weights = get_dataloader(
        "data/processed/train",
        batch_size=config["batch_size"],
        shuffle=True,
    )

    num_classes = len(train_weights)
    model = SimpleCNN(
        num_classes=num_classes,
        kernel_size=config["kernel_size"],
        pool_type=config["pool_type"],
    )
    model_name = get_name(config, balance, suffix="sanity")

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, train_loader, weights)


def kernel(balance: bool = True, pool_type: str = "max") -> None:
    base_config = get_default_config()
    base_config["pool_type"] = pool_type
    kernel_sizes = [3, 7]

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
        model = SimpleCNN(
            num_classes=num_classes,
            kernel_size=kernel_size,
            pool_type=config["pool_type"],
        )
        model_name = get_name(config, balance)

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
    parser.add_argument(
        "--pool",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="Pooling type (max or avg)",
    )
    args = parser.parse_args()

    match args.mode:
        case "sanity":
            sanity(balance=args.balance, pool_type=args.pool)
        case "defaults":
            defaults(balance=args.balance, pool_type=args.pool)
        case "kernel":
            kernel(balance=args.balance, pool_type=args.pool)
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")
