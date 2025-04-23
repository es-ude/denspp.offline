# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils

from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

#
# class Net2d(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_size = (1, 28, 28)
#         spike_grad = surrogate.FastSigmoid.apply
#
#         learn_beta= True
#         beta = 0.5
#         beta1 = 0.5
#         beta2 = 0.5
#         beta3 = 0.5
#
#
#         self.module0 = nn.Sequential(
#             #snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=False, reset_mechanism="zero"),
#             nn.Conv2d(1, 12, 5),
#             nn.MaxPool2d(2),
#             snn.Leaky(beta=beta1, spike_grad=spike_grad,learn_beta=learn_beta, init_hidden=True, reset_mechanism="zero"),
#             nn.Conv2d(12, 64, 5),
#             nn.MaxPool2d(2),
#             snn.Leaky(beta=beta2, spike_grad=spike_grad,learn_beta=learn_beta, init_hidden=True, reset_mechanism="zero"),
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 10),
#             snn.Leaky(beta=beta3, spike_grad=spike_grad,learn_beta=learn_beta, init_hidden=True, output=True, reset_mechanism="zero"))
#
#
#
# class Net1d(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_size = (1, 40)
#         spike_grad = surrogate.FastSigmoid.apply
#         beta0 = 0.5
#         beta1 = 0.5
#         beta2 = 0.5
#         beta3 = 0.5
#         beta4 = 0.5
#         beta5 = 0.5
#
#         self.module0 = nn.Sequential(
#             snn.Leaky(beta=beta0, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
#             nn.Conv1d(1, 12, 4),  # [batch_size, chanel, data-kernal_size]
#             nn.MaxPool1d(2),  # [batch_size,out_chanels, output_n/2)
#             snn.Leaky(beta=beta1, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
#             nn.Conv1d(12, 64, 4),# [batch_size, 64, data-kernal_size]
#             nn.MaxPool1d(2), # [batch_size,out_chanels, output_n/2)
#             snn.Leaky(beta=beta2, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
#             nn.Flatten(),
#             nn.Linear(448, 5),
#             #snn.Leaky(beta=beta3, spike_grad=spike_grad, init_hidden=True),
#             #nn.Linear(100, 4),
#             snn.Leaky(beta=beta5, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="zero")
#             )

class Net_h(nn.Module):
    def __init__(self):
        super().__init__()


        self.module0 = Net()
        #self.module0 = Net2()
        #self.module0 = Net_time()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.FastSigmoid.apply
        self.input_size = (1, 40)

        learn_beta = False
        reset_mechanism = 'zero'

        beta0 = 0.5
        beta1 = 0.5
        beta2 = 0.5
        beta3 = 0.5
        beta4 = 0.5
        beta5 = 0.5

        # Initialize layers
        self.lif0 = snn.Leaky(beta=beta0, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.conv1 = nn.Conv1d(1, 12, 4)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.conv2 = nn.Conv1d(12, 128, 4)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.fc1 = nn.Linear(896, 5)
        self.lif3 = snn.Leaky(beta=beta3, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)

    def forward(self, x, hidden=False):

        # Initialize hidden states and outputs at t=0
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk0, mem0 = self.lif1(x, mem0)

        cur1 = F.max_pool1d(self.conv1(spk0), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.flatten(1))
        spk3, mem3 = self.lif3(cur3, mem3)
        if hidden:
            return spk3, mem3, spk2.flatten(1), mem3.flatten(1)
        else:
            return spk3, mem3

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.FastSigmoid.apply
        self.input_size = (1, 28, 28)

        learn_beta = False
        reset_mechanism = 'zero'

        beta0 = 0.5
        beta1 = 0.5
        beta2 = 0.5
        beta3 = 0.5
        beta4 = 0.5
        beta5 = 0.5

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta3, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)

    def forward(self, x, hidden=False):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.flatten(1))
        spk3, mem3 = self.lif3(cur3, mem3)

        if hidden:
            return spk3, mem3, spk2.flatten(1), mem3.flatten(1)
        else:
            return spk3, mem3


class Net_time(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.FastSigmoid.apply
        self.input_size = (1, 40)

        learn_beta = True
        reset_mechanism = 'zero'

        beta0 = 0.5
        beta1 = 0.5
        beta2 = 0.5
        beta3 = 0.5
        beta4 = 0.5
        beta5 = 0.5

        # Initialize layers
        self.lif0 = snn.Leaky(beta=beta0, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        #self.conv1 = nn.Conv1d(1, 12, 4)
        self.fc01 = nn.Linear(1, 126)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        #self.conv2 = nn.Conv1d(12, 128, 4)
        self.fc02 = nn.Linear(63, 1024)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)
        self.fc1 = nn.Linear(512, 5)
        self.lif3 = snn.Leaky(beta=beta3, spike_grad=spike_grad, learn_beta=learn_beta, reset_mechanism=reset_mechanism)

    def forward(self, x, hidden=False):

        # Initialize hidden states and outputs at t=0
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk0, mem0 = self.lif1(x, mem0)

        cur1 = F.max_pool1d(self.fc01(spk0), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.fc02(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        if hidden:
            return spk3, mem3, spk2, mem3
        else:
            return spk3, mem3

