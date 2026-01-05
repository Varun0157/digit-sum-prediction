import os
from typing import Callable

import torch
import torch.nn as nn
import wandb

from src.data import get_dataloader
from src.loops import train_model


def run_model(
    model_fn: Callable[[int], nn.Module],
    model_name: str,
    config: dict,
    data_dir: str = "data/processed",
    ckpt_dir: str = "checkpoints",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    num_classes = len(train_weights)
    model = model_fn(num_classes)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, model_name + ".pth")

    run = wandb.init(
        project="digit-sum-prediction",
        name=model_name,
        config=config,
    )

    train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        device=device,
        ckpt_path=ckpt_path,
        class_weights=train_weights,
    )

    run.finish()
