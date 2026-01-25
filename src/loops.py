import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import confusion_matrix, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import Batch
from src.model.base import BaseModel


class EarlyStopping:
    def __init__(self, patience: int = 20) -> None:
        self.patience = patience
        self.history: list[tuple[float, float]] = []

    def update(self, val_loss: float, val_accuracy: float) -> None:
        self.history.append((val_loss, val_accuracy))

    def should_save(self) -> bool:
        if not self.history:
            return False
        current_accuracy = self.history[-1][1]
        best_accuracy = max(acc for _, acc in self.history)
        return current_accuracy == best_accuracy

    def should_stop(self) -> bool:
        if len(self.history) <= self.patience:
            return False

        recent = self.history[-self.patience :]
        before = self.history[: -self.patience]

        if not before:
            return False

        best_loss_before = min(loss for loss, _ in before)
        best_acc_before = max(acc for _, acc in before)

        recent_improved = any(
            loss < best_loss_before or acc > best_acc_before for loss, acc in recent
        )
        return not recent_improved


def _train_epoch(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    train_loss = 0.0

    for batch in tqdm(dataloader, desc="training"):
        images = batch["image"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        logits = model(images)
        loss = model.apply_criterion(logits, labels, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    num_samples = len(dataloader.dataset)
    return train_loss / num_samples


def _eval_epoch(
    model: BaseModel,
    dataloader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model, returning loss, accuracy, predictions, and labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluating"):
            images = batch["image"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            logits = model(images)
            loss = model.apply_criterion(logits, labels, criterion)
            total_loss += loss.item()

            preds = model.get_sum(logits)
            sum_labels = labels["sum"]

            correct += (preds == sum_labels).sum().item()
            total += sum_labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(sum_labels.cpu().numpy().tolist())

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model: BaseModel,
    train_dataloader: DataLoader[Batch],
    val_dataloader: DataLoader[Batch],
    num_epochs: int,
    lr: float,
    device: torch.device,
    ckpt_path: str,
    class_weights: torch.Tensor | None = None,
    patience: int = 10,
) -> None:
    print("\tTRAINING")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = _train_epoch(model, optimizer, train_dataloader, criterion, device)
        val_loss, accuracy, _, _ = _eval_epoch(model, val_dataloader, criterion, device)

        wandb.log(
            {"train/loss": train_loss, "val/loss": val_loss, "val/accuracy": accuracy}
        )

        early_stopping.update(val_loss, accuracy)

        save_model = early_stopping.should_save()
        if save_model:
            torch.save(model.state_dict(), ckpt_path)

        time_taken = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} : train loss: {train_loss:.4f}"
            f" - val loss: {val_loss:.4f} val accuracy: {accuracy:.4f}"
            f" -- time: {time_taken:.2f}s save_model: {save_model}"
        )

        if early_stopping.should_stop():
            print(f"Early stopping: no improvement for {patience} epochs")
            break


def test_model(
    model: BaseModel,
    test_dataloader: DataLoader[Batch],
    device: torch.device,
    model_name: str,
    class_weights: torch.Tensor | None = None,
    results_dir: str = "results",
) -> None:
    print("\tTESTING")

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    start_time = time.time()

    test_loss, accuracy, predictions, labels = _eval_epoch(
        model, test_dataloader, criterion, device
    )
    mae = mean_absolute_error(labels, predictions)

    num_classes = max(max(labels), max(predictions)) + 1
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    wandb.log(
        {
            "test/loss": test_loss,
            "test/accuracy": accuracy,
            "test/mae": mae,
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predictions,
                class_names=[str(i) for i in range(num_classes)],
            ),
        }
    )

    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "test_mae": float(mae),
    }
    with open(os.path.join(model_results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(model_results_dir, "confusion_matrix.npy"), cm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(model_results_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    time_taken = time.time() - start_time
    print(
        f"test results: test loss: {test_loss:.4f}"
        f" - test accuracy: {accuracy:.4f}"
        f" - MAE: {mae:.4f}"
        f" -- time: {time_taken:.2f}s"
    )
    print(f"Results saved to: {model_results_dir}")
