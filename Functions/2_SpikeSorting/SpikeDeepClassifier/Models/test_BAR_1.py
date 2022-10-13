from unittest import TestCase
import numpy as np
import torch
from BAR_1 import *

class Testbar_model(TestCase):
    def setUp(self) -> None:
        #model
        self.model = bar_model()


        size_data = 10000
        batch_size = 20
        self.test_data = torch.tensor(np.random.rand(size_data, 1, 1, 48)).float()
        self.test_data_batch = self.test_data[0:batch_size, :, :, :]



    def test_forward(self):
        x = self.test_data_batch
        out = self.model.forward(x)
        print(out)
