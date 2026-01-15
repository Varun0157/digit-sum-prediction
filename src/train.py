import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.loops import train_model


def run_model(
    model: nn.Module,
    model_name: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    ckpt_dir: str = "checkpoints",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

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
        class_weights=class_weights,
    )

    run.finish()
