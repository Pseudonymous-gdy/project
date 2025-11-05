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

class Expert_Choice(nn.Module):
    def __init__(self):
        '''
        Google: A Mixture of Experts model where each input is routed to its most suitable expert
        based on learned gating scores. The expert with the highest score is chosen for each input.
        '''
        pass
