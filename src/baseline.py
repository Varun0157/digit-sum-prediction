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
    model_fn = lambda num_classes: SimpleCNN(
        num_classes=num_classes, kernel_size=config["kernel_size"]
    )
    model_name = f"SimpleCNN_k{config['kernel_size']}_defaults"
    run_model(model_fn, model_name, config)


def kernel() -> None:
    base_config = get_default_config()
    kernel_sizes = [3, 5, 7]

    for kernel_size in kernel_sizes:
        config = {**base_config, "kernel_size": kernel_size}
        model_fn = lambda num_classes, k=kernel_size: SimpleCNN(
            num_classes=num_classes, kernel_size=k
        )
        model_name = f"SimpleCNN_k{kernel_size}"

        print(f"\n{'=' * 60}")
        print(f"Training with kernel_size={kernel_size}")
        print(f"{'=' * 60}\n")

        run_model(model_fn, model_name, config)


if __name__ == "__main__":
    defaults()
