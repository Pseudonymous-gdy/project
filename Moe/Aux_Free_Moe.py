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

class Aux_Free_Moe(nn.Module):
    def __init__(self):
        '''
        Deepseek & PKU: An Auxiliary Loss Free Mixture of Experts model.
        '''
        pass