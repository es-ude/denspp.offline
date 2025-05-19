import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.data_generator import WaveformGenerator

import matplotlib.pyplot as plt


# --- Info: Function have to start with test_*
class TestWaveformGenerator(TestCase):
    _sampling_rate: float = 20e3

    def test_result_value_length(self):
        dict = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).get_dictionary_classes()
        self.assertTrue(len(dict) > 0)

    def test_result_value_available_waveforms(self):
        dict = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).get_dictionary_classes()
        types_to_check = ['RECT', 'LIN_RISE', 'LIN_FALL', 'SAW_POS', 'SAW_NEG']
        types_checked = [True for type in types_to_check if type in dict]
        self.assertTrue(np.sum(types_checked) == len(types_to_check))


if __name__ == '__main__':
    main()
