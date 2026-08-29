"""Baseline CNN model + inference utilities.

Kept deliberately simple (per assignment: "at least one baseline model") so
it trains fast on CPU for M1, and is easy to wrap in a REST API for M2.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small 3-block CNN for 224x224 RGB binary classification (Cat=0, Dog=1)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits - apply softmax outside for probabilities


def load_model(checkpoint_path: str, device: torch.device = None) -> SimpleCNN:
    """Instantiate SimpleCNN and load trained weights from a .pt state_dict file."""
    device = device or torch.device("cpu")
    model = SimpleCNN(num_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device = None,
            class_names: list = None) -> dict:
    """Run inference on a single pre-processed image tensor.

    Args:
        model: a trained SimpleCNN (or compatible) in eval mode.
        image_tensor: a (3, H, W) or (1, 3, H, W) tensor, already normalized
            the same way as during training (see dataset.get_transforms).
        device: torch device to run on.
        class_names: optional list mapping index -> label, e.g. ["Cat", "Dog"].

    Returns:
        dict with "label", "class_index", and "probabilities" (list of floats).
        This is the shared inference utility used by both the unit tests and
        the FastAPI service in M2.
    """
    device = device or torch.device("cpu")
    class_names = class_names or ["Cat", "Dog"]

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        class_idx = int(torch.argmax(probs).item())

    return {
        "label": class_names[class_idx],
        "class_index": class_idx,
        "probabilities": probs.cpu().tolist(),
    }
