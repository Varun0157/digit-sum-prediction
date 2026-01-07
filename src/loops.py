import time

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0

    def update(self, val_loss: float, val_accuracy: float) -> None:
        improved = False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            improved = True
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            improved = True

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


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
    min_val_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = _train_epoch(model, optimizer, train_dataloader, criterion, device)
        val_loss = _test_epoch(model, val_dataloader, criterion, device)
        accuracy = _calculate_accuracy(model, val_dataloader, device)

        wandb.log(
            {"train/loss": train_loss, "val/loss": val_loss, "val/accuracy": accuracy}
        )

        save_model = val_loss < min_val_loss
        if save_model:
            min_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        early_stopping.update(val_loss, accuracy)

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
    class_weights: torch.Tensor | None = None,
) -> None:
    print("\tTESTING")

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = _get_loss_criterion(class_weights)

    start_time = time.time()

    test_loss = _test_epoch(model, test_dataloader, criterion, device)
    accuracy = _calculate_accuracy(model, test_dataloader, device)

    wandb.log({"test/loss": test_loss, "test/accuracy": accuracy})

    time_taken = time.time() - start_time
    print(
        f"test results: test loss: {test_loss:.4f}"
        + f" - test accuracy: {accuracy:.4f}"
        + f" -- time: {time_taken:.2f}s"
    )
