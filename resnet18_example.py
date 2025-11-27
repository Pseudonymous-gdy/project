"""Train and evaluate ResNet-18 on CIFAR-10 or CIFAR-100 without external configs.

Usage examples (PowerShell):
  python resnet18_example.py --dataset cifar10 --epochs 10 --batch-size 128 --ood svhn,stl10
  python resnet18_example.py --dataset cifar100 --epochs 20 --batch-size 64 --ood cifar10

This script:
 - builds a ResNet-18 (pretrained=False) and replaces the final FC to match num classes
 - trains with SGD (momentum) and a simple LR scheduler
 - saves the trained model state_dict to the specified path
 - evaluates on the in-distribution test set
 - evaluates OOD dataloaders by computing mean/max softmax confidence and reporting simple stats

Notes:
 - No configuration files are used; all options are CLI flags with reasonable defaults.
 - OOD evaluation uses the trained model; for detection experiments you can compare
   in-distribution confidence distributions vs OOD.
"""
import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# reuse dataset helpers from cifar10/cifar100
import cifar10
import cifar100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--save-path', type=str, default='./resnet18_checkpoint.pth')
    parser.add_argument('--ood', type=str, default='', help="Comma-separated OOD datasets: svhn,stl10,cifar10,cifar100")
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def get_transforms(train: bool = True):
    """Return training or test transforms used for ResNet experiments.

    Training transforms include random crop and horizontal flip. Test transforms
    are deterministic. Both normalize to mean/std = 0.5.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def build_datasets(dataset_name: str, batch_size: int, num_workers: int, download: bool = True):
    """Create train/test DataLoaders using helper modules.

    This function delegates to `cifar10.get_dataloaders` or
    `cifar100.get_dataloaders` which build torchvision datasets and
    DataLoaders. It returns (trainloader, testloader, num_classes).
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        # use cifar10.get_dataloaders from cifar10.py
        trainloader, testloader, classes = cifar10.get_dataloaders('1', batch_size=batch_size, num_workers=num_workers, data_dir='./data', download=download)
        num_classes = 10
    elif dataset_name == 'cifar100':
        # use cifar100.get_dataloaders from cifar100.py
        trainloader, testloader, classes = cifar100.get_dataloaders('1', batch_size=batch_size, num_workers=num_workers, data_dir='./data', download=download)
        num_classes = 100
    else:
        raise ValueError('Unsupported dataset')

    return trainloader, testloader, num_classes


def build_ood_loader(name: str, batch_size: int, num_workers: int, download: bool = True):
    """Return a DataLoader for the requested OOD dataset.

    Supported names: 'svhn', 'stl10', 'cifar10', 'cifar100'. The returned
    loader yields (input_tensor, label) pairs. Images are resized to 32x32 and
    normalized to the same mean/std as training.
    """
    name = name.lower()
    # Delegate to cifar100 helper when possible (keeps code consistent)
    if name in ('svhn', 'stl10', 'cifar10'):
        return cifar100.get_ood_dataloader(name, batch_size=batch_size, num_workers=num_workers, data_dir='./data', download=download)
    elif name == 'cifar100':
        # return only the test loader for CIFAR-100
        return cifar100.get_dataloaders('1', batch_size=batch_size, num_workers=num_workers, data_dir='./data', download=download)[1]
    else:
        raise ValueError(f'Unsupported OOD dataset: {name}')


def build_model(num_classes: int, device: torch.device):
    """Instantiate a ResNet-18 and replace the final FC layer.

    Args:
        num_classes: number of output classes for the final linear layer.
        device: torch.device where the model will be moved.

    Returns:
        A torch.nn.Module on the requested device.
    """
    model = resnet18(pretrained=False)
    # replace final fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch.

    Returns a tuple (avg_loss, accuracy) computed over the provided loader.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, device):
    """Evaluate model accuracy on `loader`.

    Returns the classification accuracy (float in [0,1]).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def compute_confidence_stats(model, loader, device):
    """Compute softmax max-confidence statistics over the dataset."""
    model.eval()
    confidences = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = softmax(logits)
            max_conf, _ = probs.max(dim=1)
            confidences.append(max_conf.cpu())
    if len(confidences) == 0:
        return {}
    all_conf = torch.cat(confidences).numpy()
    return {
        'mean_conf': float(all_conf.mean()),
        'std_conf': float(all_conf.std()),
        'max_conf': float(all_conf.max()),
        'min_conf': float(all_conf.min()),
        'n': int(all_conf.size),
    }


def main():
    args = parse_args()
    # device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    trainloader, testloader, num_classes = build_datasets(args.dataset, args.batch_size, args.num_workers)
    print(f'Built datasets: {args.dataset} -> num_classes={num_classes}')

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_acc = evaluate(model, testloader, device)
        scheduler.step()
        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs}  time={t1-t0:.1f}s  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}')
        # save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'test_acc': test_acc}, args.save_path)
    total_time = time.time() - start
    print(f'Training finished in {total_time/60:.2f} minutes. Best test acc: {best_acc:.4f}')

    # final evaluation on test set (reload best)
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    final_test_acc = evaluate(model, testloader, device)
    print(f'Final test accuracy: {final_test_acc:.4f}')

    # OOD evaluation
    if args.ood:
        ood_list = [s.strip() for s in args.ood.split(',') if s.strip()]
        print('\nOOD evaluation:')
        # compute in-distribution confidence stats as baseline
        id_stats = compute_confidence_stats(model, testloader, device)
        print(f'In-distribution confidence: mean={id_stats.get("mean_conf"):.4f} std={id_stats.get("std_conf"):.4f} n={id_stats.get("n")}')
        for ood_name in ood_list:
            try:
                ood_loader = build_ood_loader(ood_name, args.batch_size, args.num_workers)
                stats = compute_confidence_stats(model, ood_loader, device)
                print(f'  OOD={ood_name}: mean={stats.get("mean_conf"):.4f} std={stats.get("std_conf"):.4f} n={stats.get("n")}')
            except Exception as e:
                print(f'  Failed to build/evaluate OOD {ood_name}: {e}')


if __name__ == '__main__':
    main()
