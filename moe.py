import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

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
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        elif structure == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=pretrained)
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


class Simple_Moe(nn.Module):
    def __init__(self, num_experts=4, top_k=1, aux_loss_weight=0.01, *args, **kwargs):
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
        self.Back_bone = Backbone(*args, **kwargs)
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.noise = nn.Parameter(torch.zeros(num_experts))
        self.noise.requires_grad = True
        self.linear1 = nn.Linear(in_features=kwargs.get('num_features',32), out_features=num_experts)
        self.activation = nn.ReLU()
        pass
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