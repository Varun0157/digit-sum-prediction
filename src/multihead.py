import argparse
import os

import torch
import torchvision.transforms as T
import wandb

from src.data import get_dataloader
from src.loops import test_model
from src.model import GRUHead, MultiHeadResNet, MultiHeadSpatialAttention
from src.train import run_model


def build_augmentation() -> T.Compose:
    return T.Compose([
        T.RandomRotation(degrees=5, fill=0),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])


def get_default_config(model_name: str = "MultiHeadResNet") -> dict:
    return {
        "batch_size": 128,
        "num_epochs": 100,
        "lr": 1e-3,
        "dropout": 0.3,
        "kernel_size": 7,
        "width_multiplier": 1.0,
        "sum_loss_weight": 0.0,
        "model": model_name,
    }


def get_name(config: dict, spatial: bool = False, gru: bool = False, augment: bool = False, suffix: str = "") -> str:
    if gru:
        name = "GRUHead"
    elif spatial:
        name = "SpatialAttention"
    else:
        name = f"MultiHeadResNet_k{config['kernel_size']}"
        if config["width_multiplier"] is not None:
            name += f"_w{config['width_multiplier']:.2f}".replace(".", "")
        if config["sum_loss_weight"] is not None and config["sum_loss_weight"] > 0:
            name += f"_sum{config['sum_loss_weight']:.1f}".replace(".", "")
    if augment:
        name += "_aug"
    if suffix:
        name += f"_{suffix}"
    return name


def _build_model(
    spatial: bool,
    gru: bool,
    kernel_size: int,
    width_multiplier: float | None,
    sum_loss_weight: float | None,
    dropout: float,
) -> MultiHeadResNet | MultiHeadSpatialAttention | GRUHead:
    assert not (spatial and gru)
    assert not ((spatial or gru) and width_multiplier is not None)
    assert not ((spatial or gru) and sum_loss_weight is not None)
    if gru:
        return GRUHead(dropout=dropout)
    if spatial:
        return MultiHeadSpatialAttention(dropout=dropout)
    return MultiHeadResNet(
        kernel_size=kernel_size,
        width_multiplier=width_multiplier or 1.0,
        sum_loss_weight=sum_loss_weight or 0.0,
        dropout=dropout,
    )


def train(
    data_dir: str = "data/multi",
    kernel_size: int = 7,
    width_multiplier: float | None = None,
    sum_loss_weight: float | None = None,
    dropout: float = 0.3,
    patience: int = 20,
    spatial: bool = False,
    gru: bool = False,
    augment: bool = False,
    suffix: str = "",
) -> None:
    model = _build_model(spatial, gru, kernel_size, width_multiplier, sum_loss_weight, dropout)
    config = get_default_config(type(model).__name__)
    config.update({"kernel_size": kernel_size, "width_multiplier": width_multiplier,
                   "sum_loss_weight": sum_loss_weight, "dropout": dropout, "augment": augment})

    transform = build_augmentation() if augment else None
    train_loader, _ = get_dataloader(
        os.path.join(data_dir, "train"), batch_size=config["batch_size"], shuffle=True,
        transform=transform,
    )
    val_loader, _ = get_dataloader(
        os.path.join(data_dir, "val"), batch_size=config["batch_size"], shuffle=False,
    )

    run_model(model, get_name(config, spatial, gru, augment, suffix), config, train_loader, val_loader, patience=patience)


def eval(
    data_dir: str = "data/multi",
    split: str = "val",
    kernel_size: int = 7,
    width_multiplier: float | None = None,
    sum_loss_weight: float | None = None,
    dropout: float = 0.3,
    spatial: bool = False,
    gru: bool = False,
    suffix: str = "",
    ckpt_dir: str = "checkpoints",
) -> None:
    model = _build_model(spatial, gru, kernel_size, width_multiplier, sum_loss_weight, dropout)
    config = get_default_config(type(model).__name__)
    config.update({"kernel_size": kernel_size, "width_multiplier": width_multiplier,
                   "sum_loss_weight": sum_loss_weight, "dropout": dropout})

    model_name = get_name(config, spatial, gru, False, suffix)
    ckpt_path = os.path.join(ckpt_dir, model_name + ".pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    print(f"Loaded model from: {ckpt_path}")
    print(f"Using device: {device}")

    loader, _ = get_dataloader(
        os.path.join(data_dir, split), batch_size=config["batch_size"], shuffle=False,
    )

    run = wandb.init(
        project="digit-sum-prediction",
        name=f"eval_{model_name}_{split}",
        config=config,
        job_type="evaluation",
    )

    test_model(model=model, test_dataloader=loader, device=device, model_name=f"{model_name}_{split}")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--data_dir", type=str, default="data/multi")
    parser.add_argument("--split", type=str, default="val", help="Split for eval mode")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--width_multiplier", type=float, default=None)
    parser.add_argument("--sum_loss_weight", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--augment", action="store_true")
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
                spatial=args.spatial,
                gru=args.gru,
                augment=args.augment,
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
                spatial=args.spatial,
                gru=args.gru,
                suffix=args.suffix,
            )
