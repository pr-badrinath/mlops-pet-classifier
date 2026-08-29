"""PyTorch Dataset / DataLoader construction for the processed Cats vs Dogs data.

Assumes the folder layout produced by `data_preprocessing.py`:
    data/processed/train/Cat/*.jpg
    data/processed/train/Dog/*.jpg
    data/processed/val/...
    data/processed/test/...

This maps cleanly onto torchvision.datasets.ImageFolder, so we lean on that
rather than reinventing a custom Dataset class.
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms

# ImageNet normalization stats - standard for CNNs, even simple ones,
# since it keeps inputs in a well-behaved numeric range.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    """Return (train_transform, eval_transform).

    Train gets light data augmentation (per assignment requirement); val/test
    stay deterministic so metrics are comparable across runs.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(processed_dir: str, image_size: int = 224, batch_size: int = 32,
                     num_workers: int = 2):
    """Build train/val/test DataLoaders from the processed directory.

    Returns (train_loader, val_loader, test_loader, class_to_idx).
    """
    processed_path = Path(processed_dir)
    train_transform, eval_transform = get_transforms(image_size)

    train_ds = datasets.ImageFolder(processed_path / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(processed_path / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(processed_path / "test", transform=eval_transform)

    # persistent_workers=True keeps worker subprocesses alive across epochs
    # instead of tearing them down and respawning fresh ones every epoch.
    # Without this, each of the ~4 respawns/epoch re-imports the entire
    # stack (torch, mlflow, transformers, ...), which on Windows (spawn-based
    # multiprocessing) is a real source of wall-clock slowdown and is also
    # why you see repeated "[transformers] Disabling PyTorch..." warnings.
    # Only meaningful when num_workers > 0.
    persistent = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=persistent)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx
