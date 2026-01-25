import os

import torch
import wandb
from torch.utils.data import DataLoader

from src.data.loader import Batch
from src.loops import train_model
from src.model.base import BaseModel


def run_model(
    model: BaseModel,
    model_name: str,
    config: dict,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    class_weights: torch.Tensor | None = None,
    ckpt_dir: str = "checkpoints",
    patience: int = 10,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(ckpt_dir, exist_ok=True)
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
        patience=patience,
    )

    run.finish()
