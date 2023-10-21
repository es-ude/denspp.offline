import numpy as np

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt


class SpikingNeuralNetworks:
    def __init__(self):
        pass

    def sda_snn(self, xin: np.ndarray, y_thr: float) -> [np.ndarray, np.ndarray]:
        """Applying the spiking neural network (SNN) converter in order to extract spike pattern"""
        y_snn = np.zeros(shape=xin.size)
        y_int = np.zeros(shape=xin.size)
        y_int0 = 0.0
        gain = 1.0
        for idx, val in enumerate(np.abs(xin)):
            if y_int0 >= y_thr:
                y_int0 = 0.0
                y_snn[idx] = 1
            else:
                y_int0 += gain * val
                y_snn[idx] = 0

            y_int[idx] = y_int0

        return y_snn, y_int
