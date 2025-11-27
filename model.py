import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    Minimal neural network for hello world example.
    Input: (batch_size, 784) - flattened 28x28 images
    Output: (batch_size, 10) - 10 class logits
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 784)
        x = self.fc1(x)  # (batch_size, 128)
        x = self.relu(x)  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, 10)
        return x

