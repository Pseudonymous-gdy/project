import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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