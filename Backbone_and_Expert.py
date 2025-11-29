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


class Backbone(nn.Module):
    def __init__(self, structure='resnet18', pretrained=False, num_features=32):
        '''
        A backbone network for feature extraction.
        Args:
            structure (str): The structure of the backbone network. Currently supported: 'resnet18', 'resnet50'.
            pretrained (bool): Whether to use pretrained weights.
            num_features (int): The number of features to output from the backbone.
        '''
        super(Backbone, self).__init__()
        if structure == 'resnet18':
            weights = (
                torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.backbone = torchvision.models.resnet18(weights=weights)
        elif structure == 'resnet34':
            weights = (
                torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            self.backbone = torchvision.models.resnet34(weights=weights)
        else:
            raise ValueError("Unsupported backbone structure")

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_features)
    def forward(self, x):
        return self.backbone(x)


class Expert(nn.Module):
    def __init__(self, num_features=32, hidden_size=64, output_size=10):
        '''
        A simple expert network.

        Args:
            num_features (int): The number of input features.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
        
        Structure:
            - Linear layer with num_features input and hidden_size output
            - GELU activation function
            - Linear layer with hidden_size input and output_size output
        '''
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(num_features, hidden_size)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x