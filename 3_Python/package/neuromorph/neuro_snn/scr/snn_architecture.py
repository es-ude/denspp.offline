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


class Net_h(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.FastSigmoid.apply

        #self.module = Net()  # SpAIke
        #self.module = Net2() # mnist
        #self.module = Net_time() #spiaike serial
        #self.module= Net_time_no_conv()

        # SPaike parallel angelegt
        self.module = nn.Sequential(
                    nn.Flatten(),
                    #snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Linear(32,348),
                    #nn.Conv1d(1, 12, 4),  # [batch_size, chanel, data-kernal_size]
                    #nn.MaxPool1d(2),  # [batch_size,out_channels, output_n/2)
                    snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Linear(348, 1408),
                    #nn.Conv1d(12, 128, 4),# [batch_size, 64, data-kernal_size]
                    #nn.MaxPool1d(2), # [batch_size,out_channels, output_n/2)
                    snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Linear(1408, 5),
                    #nn.Flatten(),
                    #nn.Linear(640, 5),
                    #snn.Leaky(beta=beta3, spike_grad=spike_grad, init_hidden=True),
                    #nn.Linear(100, 4),
                    snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="zero")
                    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.FastSigmoid.apply
        self.input_size = (1, 32)

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
        self.fc1 = nn.Linear(640, 5)
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


class Net_time_no_conv(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.FastSigmoid.apply
        self.input_size = 2
        inputs = self.input_size
        hidden1 = 40
        hidden2 = 216
        hidden3 = 448
        outputs = 6

        learn_beta = True
        random_beta = False
        th_learn = True
        random_th = False
        reset_mechanism = 'zero'  # 'subtract'
        all = False

        # random start fore beta and threshold
        if random_beta:
            beta0 = 0.9
            beta1 = torch.rand(hidden1)
            beta2 = torch.rand(hidden2)
            beta3 = torch.rand(hidden3)
            beta4 = torch.rand(outputs)

        else:
            beta0 = 0.9
            beta1 = 0.9
            beta2 = 0.9
            beta3 = 0.9
            beta4 = 0.9

        if random_th:
            th0 = torch.rand(inputs)
            th1 = torch.rand(hidden1)
            th2 = torch.rand(hidden2)
            th3 = torch.rand(hidden3)
            th4 = torch.rand(outputs)
        else:
            th0 = 1.0
            th1 = 1.0
            th2 = 1.0
            th3 = 1.0
            th4 = 1.0

        Vi=torch.rand(inputs)
        V0=torch.rand(hidden1)
        V1=torch.rand(hidden2)
        V2=torch.rand(hidden3)
        V3=torch.rand(outputs)

        # Net
        #self.fc00 = nn.Linear(1, inputs)
        self.lif00 = snn.Leaky(beta=1., threshold=torch.rand(inputs), spike_grad=spike_grad,
                               learn_threshold=th_learn, reset_mechanism=reset_mechanism, init_hidden=True)

        #self.lif00 = snn.RLeaky(beta=beta0, threshold=th0, V=Vi, all_to_all=all, spike_grad=spike_grad,
        #                       #linear_features=inputs,
        #                       learn_beta=learn_beta, learn_threshold=th_learn,
        #                       init_hidden=True,
        #                       inhibition=inhibitation,
        #                       reset_mechanism=reset_mechanism)
        self.fc0 = nn.Linear(inputs, hidden1)
        #self.lif0 = snn.Leaky(beta=beta1, threshold=th1, spike_grad=spike_grad, learn_beta=learn_beta,
        #                      learn_threshold=th_learn, reset_mechanism=reset_mechanism, init_hidden=True)
        self.lif0 = snn.RLeaky(beta=beta1, threshold=th1, V=V0, all_to_all=all, spike_grad=spike_grad,
                               #linear_features=hidden1,
                               learn_beta=learn_beta, learn_threshold=th_learn,
                               init_hidden=True,
                               #inhibition=inhibitation,
                               reset_mechanism=reset_mechanism)
        self.fc1 = nn.Linear(hidden1, hidden2)  #
        #self.lif1 = snn.Leaky(beta=beta2, threshold=th2, spike_grad=spike_grad, learn_beta=learn_beta,
        #                      learn_threshold=th_learn, reset_mechanism=reset_mechanism, init_hidden=True)
        self.lif1 = snn.RLeaky(beta=beta2, threshold=th2, V=V1, all_to_all=all, spike_grad=spike_grad,
                               #linear_features=hidden2,
                               learn_beta=learn_beta, learn_threshold=th_learn,
                               init_hidden=True,
                               #inhibition=inhibitation,
                               reset_mechanism=reset_mechanism)
        #self.do1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden2, hidden3)
        #self.lif2 = snn.Leaky(beta=beta3, threshold=th3, spike_grad=spike_grad,learn_beta=learn_beta,
        #                      learn_threshold=th_learn, reset_mechanism=reset_mechanism, init_hidden=True)
        self.lif2 = snn.RLeaky(beta=beta3, threshold=th3, V=V2, all_to_all=all, spike_grad=spike_grad,
                               #linear_features=hidden3,
                               learn_beta=learn_beta, learn_threshold=th_learn,
                               init_hidden=True,
                               #inhibition=inhibitation,
                               reset_mechanism=reset_mechanism)
        #self.do2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden3, outputs)
        #self.lif3 = snn.Leaky(beta=beta4, threshold=th4, spike_grad=spike_grad, learn_beta=learn_beta,
        #                      learn_threshold=th_learn, reset_mechanism=reset_mechanism, init_hidden=True, output=True)
        self.lif3 = snn.RLeaky(beta=beta4, threshold=th4, V=V3, all_to_all=all, spike_grad=spike_grad,
                               #linear_features=outputs,
                               learn_beta=learn_beta, learn_threshold=th_learn,
                               init_hidden=True, output=True,
                               #inhibition=inhibitation,
                               reset_mechanism=reset_mechanism)


    def forward(self, data_in, hidden=False):
        spk = self.lif00(data_in)

        cur0 = self.fc0(spk)  # (spk00)  #
        #spk0, mem0 = self.lif0(cur0, mem0)
        spk0 = self.lif0(cur0)#, spk0, mem0)

        cur1 = self.fc1(spk0)
        #spk1, mem1 = self.lif1(cur1, mem1)
        spk1 = self.lif1(cur1)#, spk1, mem1)

        cur2 = self.fc2(spk1)
        #spk2, mem2 = self.lif2(cur2, mem2)
        spk2 = self.lif2(cur2)#, spk2,mem2)

        cur3 = self.fc3(spk2)
        #spk3, mem3 = self.lif3(cur3, mem3)
        spk3, mem3 = self.lif3(cur3)#, spk3,mem3)
        #print(f'spk_3:{spk3.size()}')

        if hidden:
            return spk3.flatten(1), mem3.flatten(1), spk2.flatten(1),spk1.flatten(1),spk0.flatten(1), spk.flatten(1)
        else:
            return spk3.flatten(1), mem3.flatten(1)
