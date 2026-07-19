"""
CNN classifiers for Galaxy10.

- ResNet50 transfer learning: Lab 2 (main experimental model)
- SimpleCNN: lightweight LeNet-style CNN (Lab 1 spirit) for fast smoke tests
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50_classifier(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet50 classifier head for Galaxy10 (10 morphology classes).
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze all backbone parameters (Lab 2 pattern).
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def unfreeze_layer4_and_fc(model: nn.Module) -> None:
    """Enable fine-tuning of the last residual block and classifier head."""
    for name, param in model.named_parameters():
        if "layer4" in name or name.startswith("fc"):
            param.requires_grad = True


class SimpleCNN(nn.Module):
    """
    Small CNN for smoke / CPU sanity runs.

    Inspired by the module's early CNN labs (conv → pool → FC), sized for
    64x64 RGB inputs so end-to-end tests finish quickly on CPU.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_simple_cnn(num_classes: int = 10) -> nn.Module:
    """Factory for the lightweight smoke-test CNN."""
    return SimpleCNN(num_classes=num_classes)
