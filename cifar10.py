"""CIFAR-10 dataset helpers and two preprocessing pipelines.

This module provides small utility functions used by example training scripts.
It offers:
 - get_transforms(setting): return a torchvision transform pipeline
 - get_dataloaders(...): create train/test DataLoaders for CIFAR-10

Usage notes / contract:
 - get_transforms accepts '1' or 'basic' for a minimal pipeline (ToTensor + Normalize),
   and '2' or 'aug' for a common augmentation pipeline (RandomCrop, Flip, ColorJitter).
 - get_dataloaders returns a tuple (trainloader, testloader, classes). Each loader
   yields (tensor, label) pairs where tensor shape is [B, 3, 32, 32].
 - By default this module creates two convenience module-level dataloaders
   (`trainloader1/testloader1` and `trainloader2/testloader2`) for interactive use.
   BEWARE: importing this module will create those dataloaders and may trigger
   dataset downloads (side effect). If you want to avoid download-on-import, call
   `get_dataloaders(...)` explicitly instead of relying on the module variables.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet18
import os
import numpy as np


# ---
# Data transforms and helpers for two experimental settings
# Setting 1 (basic): only tensor conversion and normalization
# Setting 2 (augmented): random crop, flip, color jitter + normalization
def get_transforms(setting: str = '1'):
    """Return a torchvision transform pipeline for the given setting.

    Args:
        setting: '1' or 'basic' returns only ToTensor + Normalize.
                 '2' or 'aug' returns a stronger training-time augmentation pipeline.

    Returns:
        torchvision.transforms.Compose: composed transform to apply to PIL images.
    """
    if setting in ('1', 'basic'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if setting in ('2', 'aug'):
        return transforms.Compose([
            # data augmentation commonly used for CIFAR
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    raise ValueError(f"Unknown transform setting: {setting}")


def get_dataloaders(setting: str = '1', batch_size: int = 64, num_workers: int = 2,
                    data_dir: str = './data', download: bool = True):
    """Create CIFAR10 train/test dataloaders for the chosen transform setting.

    Args:
        setting: transform setting passed to `get_transforms`.
        batch_size: samples per batch.
        num_workers: DataLoader worker processes.
        data_dir: root directory to store/download datasets.
        download: whether to download CIFAR-10 if missing.

    Returns:
        (trainloader, testloader, classes)
          - trainloader/testloader: torch.utils.data.DataLoader
          - classes: tuple of 10 string labels for CIFAR-10
    """
    train_transform = get_transforms(setting)
    # keep test transform deterministic (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=download, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=download, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


# Create two named dataloader sets for convenience / backward compatibility
# You can call get_dataloaders(...) directly from other scripts instead of
# relying on module-level variables. These module-level loaders are useful for
# quick interactive experiments but cause a dataset download on import if files
# are missing.
transform1 = get_transforms('1')
trainloader1, testloader1, classes = get_dataloaders('1', batch_size=64, num_workers=2, data_dir='./data')

transform2 = get_transforms('2')
trainloader2, testloader2, _ = get_dataloaders('2', batch_size=64, num_workers=2, data_dir='./data')