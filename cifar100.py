"""CIFAR-100 dataset helpers and OOD dataloader utilities.

This module mirrors the CIFAR-10 helpers but is specialized for CIFAR-100.
It exposes factory functions to obtain train/test DataLoaders and an OOD
data loader helper for evaluating model robustness.

Important:
 - get_dataloaders returns (trainloader, testloader, classes). `classes` is
   a list of numeric labels [0..99] representing the 100 fine-grained classes.
 - get_ood_dataloader supports a small set of off-distribution datasets
   (SVHN, STL10, CIFAR-10). Returned loaders yield (tensor, label) pairs; labels
   for OOD datasets are available but typically not used for detection metrics.

Usage example:
    train_loader, test_loader, classes = get_dataloaders('1', batch_size=64)
    svhn_loader = get_ood_dataloader('svhn', batch_size=128)
"""

import torch
import torchvision
import torchvision.transforms as transforms


# ---
# CIFAR-100: in-distribution settings and OOD helpers
# Two settings: '1' basic normalization, '2' augmented training transforms
def get_transforms(setting: str = '1'):
    """Return transform pipeline for CIFAR-100 experiments.

    Args:
        setting: '1' or 'basic' for minimal pipeline; '2' or 'aug' for training augmentations.

    Returns:
        torchvision.transforms.Compose instance.
    """
    if setting in ('1', 'basic'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if setting in ('2', 'aug'):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    raise ValueError(f"Unknown transform setting: {setting}")


def get_dataloaders(setting: str = '1', batch_size: int = 64, num_workers: int = 2,
                    data_dir: str = './data', download: bool = True):
    """Create CIFAR-100 train/test dataloaders for the chosen transform setting.

    Args:
        setting: transform selection (see get_transforms).
        batch_size: batch size for loaders.
        num_workers: number of worker processes.
        data_dir: dataset root directory.
        download: whether to download the dataset if missing.

    Returns:
        (trainloader, testloader, classes)
    """
    train_transform = get_transforms(setting)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                             download=download, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                            download=download, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    # CIFAR-100 fine labels are 100 classes; return a list of numeric labels by default
    classes = list(range(100))

    return trainloader, testloader, classes


def get_ood_dataloader(ood_name: str = 'svhn', batch_size: int = 64, num_workers: int = 2,
                       data_dir: str = './data', download: bool = True):
    """Return a dataloader for an OOD dataset to evaluate robustness/stability.

    Supported ood_name: 'svhn', 'stl10', 'cifar10'
    - SVHN: street view house numbers (32x32) - good OOD for CIFAR
    - STL10: larger images (will be resized to 32x32)
    - CIFAR10: CIFAR-10 as a different-distribution test (labels differ)

    Returns:
        torch.utils.data.DataLoader for the requested OOD dataset.
    """
    ood_name = ood_name.lower()
    # shared test transform (resize if needed -> to 32x32, deterministic)
    base_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if ood_name == 'svhn':
        # SVHN has different constructor: split='test'
        ds = torchvision.datasets.SVHN(root=data_dir, split='test', download=download, transform=base_transform)
    elif ood_name == 'stl10':
        ds = torchvision.datasets.STL10(root=data_dir, split='test', download=download, transform=base_transform)
    elif ood_name == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=base_transform)
    else:
        raise ValueError(f"Unsupported OOD dataset: {ood_name}. Supported: svhn, stl10, cifar10")

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


# Module-level convenience variables (basic and augmented) for quick import
# Creating these variables triggers dataset construction/download on import.
transform100_basic = get_transforms('1')
trainloader100_basic, testloader100_basic, classes100 = get_dataloaders('1', batch_size=64, num_workers=2, data_dir='./data')

transform100_aug = get_transforms('2')
trainloader100_aug, testloader100_aug, _ = get_dataloaders('2', batch_size=64, num_workers=2, data_dir='./data')
