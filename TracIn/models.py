import torch
import torch.nn as nn


class MLP2layers(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(MLP2layers, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        return self.fc(x)
