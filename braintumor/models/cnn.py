import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            x = F.relu(self.conv1(dummy_input))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()  # total features after conv+pool

        self.fc1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
