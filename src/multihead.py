import argparse
import os

import torch
import wandb

from src.data import get_dataloader
from src.loops import test_model
from src.model import MultiHeadResNet
from src.train import run_model


def get_default_config() -> dict:
    return {
        "batch_size": 128,
        "num_epochs": 100,
        "lr": 1e-3,
        "dropout": 0.3,
        "kernel_size": 7,
        "width_multiplier": 1.0,
        "sum_loss_weight": 0.0,
        "model": "MultiHeadResNet",
    }


def get_name(config: dict, suffix: str = "") -> str:
    name = f"MultiHeadResNet_k{config['kernel_size']}"
    if config["width_multiplier"] != 1.0:
        name += f"_w{config['width_multiplier']:.2f}".replace(".", "")
    if config["sum_loss_weight"] > 0:
        name += f"_sum{config['sum_loss_weight']:.1f}".replace(".", "")
    if suffix:
        name += f"_{suffix}"
    return name


def train(
    data_dir: str = "data/multi",
    kernel_size: int = 7,
    width_multiplier: float = 1.0,
    sum_loss_weight: float = 0.0,
    dropout: float = 0.3,
    patience: int = 10,
    suffix: str = "",
) -> None:
    config = get_default_config()
    config["kernel_size"] = kernel_size
    config["width_multiplier"] = width_multiplier
    config["sum_loss_weight"] = sum_loss_weight
    config["dropout"] = dropout

    train_loader, train_weights = get_dataloader(
        os.path.join(data_dir, "train"),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader, _ = get_dataloader(
        os.path.join(data_dir, "val"),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = MultiHeadResNet(
        kernel_size=kernel_size,
        width_multiplier=width_multiplier,
        sum_loss_weight=sum_loss_weight,
        dropout=dropout,
    )
    model_name = get_name(config, suffix)

    run_model(model, model_name, config, train_loader, val_loader, patience=patience)


def eval(
    data_dir: str = "data/multi",
    split: str = "val",
    kernel_size: int = 7,
    width_multiplier: float = 1.0,
    sum_loss_weight: float = 0.0,
    dropout: float = 0.3,
    suffix: str = "",
    ckpt_dir: str = "checkpoints",
) -> None:
    config = get_default_config()
    config["kernel_size"] = kernel_size
    config["width_multiplier"] = width_multiplier
    config["sum_loss_weight"] = sum_loss_weight
    config["dropout"] = dropout

    loader, _ = get_dataloader(
        os.path.join(data_dir, split),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = MultiHeadResNet(
        kernel_size=kernel_size,
        width_multiplier=width_multiplier,
        sum_loss_weight=sum_loss_weight,
        dropout=dropout,
    )
    model_name = get_name(config, suffix)
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
        name=f"eval_{model_name}_{split}",
        config=config,
        job_type="evaluation",
    )

    test_model(
        model=model,
        test_dataloader=loader,
        device=device,
        model_name=f"{model_name}_{split}",
    )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument("--data_dir", type=str, default="data/multi")
    parser.add_argument("--split", type=str, default="val", help="Split for eval mode")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--width_multiplier", type=float, default=1.0)
    parser.add_argument("--sum_loss_weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    match args.mode:
        case "train":
            train(
                data_dir=args.data_dir,
                kernel_size=args.kernel_size,
                width_multiplier=args.width_multiplier,
                sum_loss_weight=args.sum_loss_weight,
                dropout=args.dropout,
                patience=args.patience,
                suffix=args.suffix,
            )
        case "eval":
            eval(
                data_dir=args.data_dir,
                split=args.split,
                kernel_size=args.kernel_size,
                width_multiplier=args.width_multiplier,
                sum_loss_weight=args.sum_loss_weight,
                dropout=args.dropout,
                suffix=args.suffix,
            )
