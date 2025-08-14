import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.data_call.waveform_dataset import SettingsWaveformDataset, build_waveform_dataset


settings = SettingsWaveformDataset(
    wfg_type=['RECT_HALF', 'LIN_RISE', 'LIN_FALL', 'SINE_HALF', 'SINE_HALF_INV', 'SINE_FULL', 'TRI_HALF', 'TRI_FULL', 'SAW_POS', 'SAW_NEG', 'GAUSS'],
    wfg_freq=[1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2],
    num_samples=2,
    time_idle=20,
    scale_amp=1.0,
    sampling_rate=20e3,
    noise_add=True,
    noise_pwr_db=-30.0,
    do_normalize=False
)


# --- Info: Function have to start with test_*
class TestWaveformDataset(TestCase):
    def test_build_example(self):
        set_test = deepcopy(settings)
        dataset = build_waveform_dataset(set_test)
        self.assertTrue([True for key in dataset.keys() if key in ['data', 'label', 'dict', 'fs']])

    def test_build_example_size(self):
        set_test = deepcopy(settings)
        dataset = build_waveform_dataset(set_test)
        self.assertTrue(dataset['data'].shape[0] == dataset['label'].size)

    def test_build_example_label(self):
        set_test = deepcopy(settings)
        dataset = build_waveform_dataset(set_test)
        self.assertEqual(dataset['dict'], set_test.wfg_type)

    def test_build_different_freq(self):
        set_test = deepcopy(settings)
        set_test.wfg_type = ['RECT_HALF', 'LIN_RISE']
        set_test.wfg_freq = [1e2, 2e2]

        dataset = build_waveform_dataset(set_test)
        self.assertEqual(dataset['dict'], set_test.wfg_type)


if __name__ == '__main__':
    main()
