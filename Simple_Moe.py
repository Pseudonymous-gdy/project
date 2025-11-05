import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import cifar10
import cifar100
import math
from torch.utils.data import TensorDataset, DataLoader, Subset
from Backbone_and_Expert import *

class Simple_Moe(nn.Module):
    def __init__(self, num_experts=4, top_k=1, aux_loss_weight=0.01, **kwargs):
        '''
        Google: A Simple Mixture of Experts model with:
            - A gating network that outputs a distribution over experts
            - Top-k selection of experts based on gating scores
            - Auxiliary loss to encourage diversity in gating scores
        '''
        super(Simple_Moe, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.Back_bone = Backbone(structure=kwargs.get('backbone_structure','resnet18'),
                                  pretrained=kwargs.get('backbone_pretrained',False),
                                  num_features=kwargs.get('num_features',32))
        self.experts = nn.ModuleList([Expert(output_size=kwargs.get('output_size',10)) for _ in range(num_experts)])
        self.noise = nn.Parameter(torch.zeros(num_experts))
        self.noise.requires_grad = True
        self.linear1 = nn.Linear(in_features=kwargs.get('num_features',32), out_features=num_experts)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        '''
        features = self.Back_bone(x)
        # pass the gate
        gate_scores = self.linear1(features)
        # add noise following a Gaussian distribution
        gate_scores = gate_scores + self.noise * torch.randn_like(gate_scores)
        # get the top-k experts
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=1)
        # get softmax scores for the top-k experts
        topk_softmax = F.softmax(topk_scores, dim=1)
        # compute outputs for all experts: shape (batch, num_experts, output_size)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        # select the top-k expert outputs per sample using gather
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        topk_outputs = torch.gather(expert_outputs, dim=1, index=idx_expanded)
        # combine the outputs of the top-k experts
        output = torch.sum(topk_outputs * topk_softmax.unsqueeze(-1), dim=1)
        return output

    def train_one_epoch(self, loader, optimizer, device, criterion=None, max_batches=None):
        """Train the Simple_Moe for one epoch on the provided DataLoader.

        Args:
            loader: a DataLoader yielding (inputs, targets)
            optimizer: optimizer instance for model parameters
            device: torch.device ('cpu' or 'cuda')
            criterion: optional loss function (default: CrossEntropyLoss)

        Returns:
            tuple: (avg_loss, accuracy) for the epoch
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # forward pass to get expert-mixed output
            outputs = self.forward(inputs)

            loss = criterion(outputs, targets)

            # auxiliary gating loss (encourage balanced expert usage)
            # compute gate probs from backbone features
            features = self.Back_bone(inputs)
            gate_scores = self.linear1(features)
            gate_scores = gate_scores + self.noise * torch.randn_like(gate_scores)
            gate_probs = F.softmax(gate_scores, dim=1)
            mean_gate = gate_probs.mean(dim=0)  # shape: (num_experts,)
            # KL(mean_gate || uniform) = sum mean_gate * (log mean_gate - log 1/K)
            K = float(self.num_experts)
            kl_uniform = (mean_gate * (torch.log(mean_gate + 1e-12) - math.log(1.0 / K))).sum()

            total_loss = loss + self.aux_loss_weight * kl_uniform

            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item()) * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def evaluate(self, loader, device, max_batches=None):
        """Evaluate model accuracy on `loader`.

        Args:
            loader: DataLoader yielding (inputs, targets)
            device: torch.device
            max_batches: optionally limit number of batches to evaluate (for fast unit tests)

        Returns:
            accuracy (float)
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.forward(inputs)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        return correct / total if total > 0 else 0.0


class Expert_Choice(nn.Module):
    def __init__(self):
        '''
        Google: A Mixture of Experts model where each input is routed to its most suitable expert
        based on learned gating scores. The expert with the highest score is chosen for each input.
        '''
        pass


class Hierarchical_Moe(nn.Module):
    def __init__(self):
        '''
        Meta & MIT: A Hierarchical Mixture of Experts model with:
            - Hierachies of experts at multiple levels
            - Gating networks at each level to route inputs to appropriate experts
        '''
        pass


class Soft_Moe(nn.Module):
    def __init__(self):
        '''
        Google: A Soft Mixture of Experts model where all experts contribute to the output
        weighted by their gating scores, allowing for smoother expert utilization.
        '''
        pass

class Bayesian_NN_Moe(nn.Module):
    def __init__(self):
        '''
        Our model: A Bayesian Neural Network Mixture of Experts model with:
            - Gating network that outputs a distribution over experts
            - Varied inference techniques including:
                - Considering Mass Coverage of Expert Probabilities
                - Ordinary Top-k Selection with Auxiliary Loss
        '''
        pass


if __name__ == '__main__':
    # Basic unit test for Simple_Moe.train_one_epoch
    print('Running basic unit test for Simple_Moe...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run a short diagnostic experiment with settings that are easier to train
    # Changes for the diagnostic run:
    # - use soft mixing (top_k=num_experts) so all experts participate
    # - disable auxiliary KL loss to isolate classification signal
    # - larger batch and more batches per epoch for stronger gradients
    # - smaller learning rate for Adam
    EPOCHS = 100
    # create a Simple_Moe instance with soft mixing and no aux loss for debugging
    model = Simple_Moe(num_experts=3, top_k=1, aux_loss_weight=0.0, num_features=32, output_size=100).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    # increase batch size and process more batches per epoch to strengthen training signal
    train_loader, test_loader, _ = cifar100.get_dataloaders('1', batch_size=64, num_workers=0, data_dir='./data', download=False)
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optim, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f'cifar100 quick train: avg_loss={avg_loss:.4f}, train_acc={acc:.4f}, test_acc={test_acc:.4f}')