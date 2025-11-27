import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

# Your repo provides these modules; keep usage consistent with your project layout.
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert
from Moe.Simple_Moe import Simple_Moe
from Moe.Aux_Free_Moe import Aux_Free_Moe
from Moe.BASE_Moe import BASE_Moe
from Moe.Bayesian_NN_Moe import Bayesian_NN_Moe
from Moe.Expert_Choice_Moe import Expert_Choice

'''
This file is used to test the training time and accuracy of different 
Mixture of Experts (MoE) models on CIFAR-10 and CIFAR-100 datasets.
Set the arguments to test different models.
'''
