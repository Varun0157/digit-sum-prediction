from src.data import get_dataloader
from src.model import SimpleCNN
from src.train import run_model


def get_default_config() -> dict:
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 1e-3,
        "kernel_size": 5,
        "model": "SimpleCNN",
    }


def defaults() -> None:
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
    model_name = f"SimpleCNN_k{config['kernel_size']}_defaults"

    run_model(model, model_name, config, train_loader, val_loader, train_weights)


def sanity() -> None:
    config = get_default_config()
    config["num_epochs"] = 10

    train_loader, train_weights = get_dataloader(
        "data/processed/train",
        batch_size=config["batch_size"],
        shuffle=True,
    )

    num_classes = len(train_weights)
    model = SimpleCNN(num_classes=num_classes, kernel_size=config["kernel_size"])
    model_name = "SimpleCNN_sanity_check"

    run_model(model, model_name, config, train_loader, train_loader, train_weights)


def kernel() -> None:
    base_config = get_default_config()
    kernel_sizes = [3, 5, 7]

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
        model_name = f"SimpleCNN_k{kernel_size}"

        print(f"\n{'=' * 60}")
        print(f"Training with kernel_size={kernel_size}")
        print(f"{'=' * 60}\n")

        run_model(model, model_name, config, train_loader, val_loader, train_weights)


if __name__ == "__main__":
    sanity()
