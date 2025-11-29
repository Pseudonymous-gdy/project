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

def main():
    train_loader, test_loader, classes = cifar100.get_dataloaders(setting='2', batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive number of classes from the dataloader metadata
    num_classes = len(classes) if hasattr(classes, "__len__") else int(classes)

    models = [
        Simple_Moe(output_size=num_classes).to(device),
        Aux_Free_Moe(output_size=num_classes).to(device),
        BASE_Moe(output_size=num_classes).to(device),
        Bayesian_NN_Moe(output_size=num_classes).to(device),
        Expert_Choice(output_size=num_classes).to(device),
    ]

    for model in models:
        print(model.__class__.__name__)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        for i in range(40):
            model.train_one_epoch(train_loader, optimizer, device)
            if i % 10 == 0:
                output = model.evaluate(train_loader, device)
                print('Using Device:', device)
                print('Model Accuracy:', output)
        print()
        print(model.evaluate(test_loader, device))

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()