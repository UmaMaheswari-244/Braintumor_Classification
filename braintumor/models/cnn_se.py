# models/cnn_se.py
import torch
import torch.nn as nn
from models.se import SEBlock

class SimpleCNN_SE(nn.Module):
    """
    A simple CNN with SE block after the last conv block.
    Mirrors the SimpleCNN_CBAM but uses SE.
    """
    def __init__(self, num_classes: int, img_size: int = 224):
        super(SimpleCNN_SE, self).__init__()

        # ===== Convolutional trunk =====
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2)  # /2
  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2)  # /4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(2)  # /8

        # Last conv marker (for Grad-CAM)
        self.last_conv = self.conv3

        # ===== SE block after last conv =====
        self.se = SEBlock(128)

        # ===== Classifier =====
        feat_hw = img_size // 8
        in_features = 128 * feat_hw * feat_hw

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu_fc1 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3 (last conv)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # SE attention
        x = self.se(x)

        # Classifier head
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
