import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class RoadSignOcclusionCNN(nn.Module):
    """Shared CNN backbone with sign-classification and occlusion heads."""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Dropout(0.30))
        self.sign_head = nn.Linear(192, num_classes)
        self.occlusion_head = nn.Linear(192, 2)

    def forward(self, x):
        features = self.shared(self.features(x))
        return self.sign_head(features), self.occlusion_head(features)
