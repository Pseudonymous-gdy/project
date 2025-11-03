import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

class Expert(nn.Module):
    def __init__(self):
        pass


class Simple_Moe(nn.Module):
    def __init__(self):
        '''
        Google: A Simple Mixture of Experts model with:
        - A gating network that outputs a distribution over experts
        - Top-k selection of experts based on gating scores
        - Auxiliary loss to encourage diversity in gating scores
        '''
        pass


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