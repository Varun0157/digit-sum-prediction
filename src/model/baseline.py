import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple baseline CNN for digit sum prediction.
    ref: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self, num_classes: int = 37, kernel_size: int = 5) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size)

        # Calculate spatial dimensions after conv+pool layers
        # Input: (40, 168)
        # Conv1: (in - k + 1) -> Pool1: // 2 -> Conv2: (in - k + 1) -> Pool2: // 2
        # h_out = ((40 - k + 1) // 2 - k + 1) // 2
        # w_out = ((168 - k + 1) // 2 - k + 1) // 2
        h_out = ((40 - kernel_size + 1) // 2 - kernel_size + 1) // 2
        w_out = ((168 - kernel_size + 1) // 2 - kernel_size + 1) // 2
        fc1_input_size = 64 * h_out * w_out

        self.fc1 = nn.Linear(fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
