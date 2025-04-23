import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import torch
import torch.nn as nn
import torch.utils as data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary

# darstellung und sonstige
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


import numpy as np
import io
import copy
import itertools
import os
import time


def save_net(net, path):
    if net.device != "cpu":
        net.module.cpu()
        PROJECT_PATH = os.path.abspath(".")
        rel_path = os.path.join(PROJECT_PATH, path)
        if not os.path.exists(rel_path):
            os.makedirs(rel_path)
        print(rel_path)
        torch.save(net, "./"+path+net.get_name()+'.pt')

def load_net(path, file_name):
    com_path= path+file_name+'.pt'
    if not os.path.exists(com_path):
        raise Exception(f'model does not exist: {com_path}')
    else:
        return torch.load(com_path)