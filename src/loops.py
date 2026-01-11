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


def _get_loss_criterion(weights: torch.Tensor | None = None) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=weights, reduction="sum")


def _calculate_accuracy(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="calculating accuracy"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def _get_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="collecting predictions"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return all_predictions, all_labels


def _test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    num_samples = len(dataloader.dataset)
    return test_loss / num_samples


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(dataloader, desc="training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    num_samples = len(dataloader.dataset)
    return train_loss / num_samples


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
    criterion = _get_loss_criterion(class_weights)

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = _train_epoch(model, optimizer, train_dataloader, criterion, device)
        val_loss = _test_epoch(model, val_dataloader, criterion, device)
        accuracy = _calculate_accuracy(model, val_dataloader, device)

        wandb.log(
            {"train/loss": train_loss, "val/loss": val_loss, "val/accuracy": accuracy}
        )

        early_stopping.update(val_loss, accuracy)

        save_model = early_stopping.should_save()
        if save_model:
            torch.save(model.state_dict(), ckpt_path)

        time_taken = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} : train Loss: {train_loss:.4f}"
            + f" - val loss: {val_loss:.4f} val accuracy: {accuracy:.4f} "
            + f"-- time: {time_taken:.2f}s save_model: {save_model}"
        )

        if early_stopping.should_stop():
            print(f"Early stopping: no improvement for {patience} epochs")
            break


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    model_name: str,
    class_weights: torch.Tensor | None = None,
    results_dir: str = "results",
) -> None:
    print("\tTESTING")

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = _get_loss_criterion(class_weights)

    start_time = time.time()

    test_loss = _test_epoch(model, test_dataloader, criterion, device)
    accuracy = _calculate_accuracy(model, test_dataloader, device)
    predictions, labels = _get_predictions(model, test_dataloader, device)

    mae = mean_absolute_error(labels, predictions)

    # Get number of classes from predictions/labels range
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
        + f" - test accuracy: {accuracy:.4f}"
        + f" - MAE: {mae:.4f}"
        + f" -- time: {time_taken:.2f}s"
    )
    print(f"Results saved to: {model_results_dir}")
