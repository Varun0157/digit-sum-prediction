import argparse
import os

import torch
import wandb

from src.data import get_dataloader
from src.loops import test_model
from src.model import MultiBranchCNN
from src.train import run_model


def get_default_config() -> dict:
    return {
        "batch_size": 128,
        "num_epochs": 200,
        "lr": 5e-4,
        "kernel_sizes": [5, 7],
        "pool_type": "avg",
        "model": "MultiBranchCNN",
    }


def get_name(config: dict, balance: bool, suffix: str = "") -> str:
    weight_mode = "balanced" if balance else "unweighted"
    pool_type = config.get("pool_type", "avg")
    kernel_str = "_".join(map(str, config["kernel_sizes"]))
    name = f"MultiBranchCNN_k{kernel_str}_{pool_type}_{weight_mode}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def train(balance: bool = False, pool_type: str = "avg") -> None:
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
    model = MultiBranchCNN(
        num_classes=num_classes,
        kernel_sizes=config["kernel_sizes"],
        pool_type=config["pool_type"],
    )
    model_name = get_name(config, balance)

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, val_loader, weights)


def sanity(balance: bool = False, pool_type: str = "avg") -> None:
    config = get_default_config()
    config["num_epochs"] = 200
    config["pool_type"] = pool_type

    train_loader, train_weights = get_dataloader(
        "data/processed/train",
        batch_size=config["batch_size"],
        shuffle=True,
    )

    num_classes = len(train_weights)
    model = MultiBranchCNN(
        num_classes=num_classes,
        kernel_sizes=config["kernel_sizes"],
        pool_type=config["pool_type"],
    )
    model_name = get_name(config, balance, suffix="sanity")

    weights = train_weights if balance else None
    run_model(model, model_name, config, train_loader, train_loader, weights)


def eval(
    balance: bool = False,
    pool_type: str = "avg",
    kernel_sizes: list[int] | None = None,
    ckpt_dir: str = "checkpoints",
) -> None:
    config = get_default_config()
    config["pool_type"] = pool_type
    if kernel_sizes is not None:
        config["kernel_sizes"] = kernel_sizes

    val_loader, val_weights = get_dataloader(
        "data/processed/val",
        batch_size=config["batch_size"],
        shuffle=False,
    )

    num_classes = len(val_weights)
    model = MultiBranchCNN(
        num_classes=num_classes,
        kernel_sizes=config["kernel_sizes"],
        pool_type=config["pool_type"],
    )
    model_name = get_name(config, balance)
    ckpt_path = os.path.join(ckpt_dir, model_name + ".pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    print(f"Loaded model from: {ckpt_path}")
    print(f"Using device: {device}")

    run = wandb.init(
        project="digit-sum-prediction",
        name=f"eval_{model_name}",
        config=config,
        job_type="evaluation",
    )

    weights = val_weights if balance else None
    test_model(
        model=model,
        test_dataloader=val_loader,
        device=device,
        model_name=model_name,
        class_weights=weights,
    )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["sanity", "train", "eval"],
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
        default="avg",
        choices=["max", "avg"],
        help="Pooling type (max or avg)",
    )
    parser.add_argument(
        "--kernels",
        type=int,
        nargs="+",
        default=None,
        help="Kernel sizes for branches (for eval mode, overrides default)",
    )
    args = parser.parse_args()

    match args.mode:
        case "sanity":
            sanity(balance=args.balance, pool_type=args.pool)
        case "train":
            train(balance=args.balance, pool_type=args.pool)
        case "eval":
            eval(
                balance=args.balance,
                pool_type=args.pool,
                kernel_sizes=args.kernels,
            )
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")
