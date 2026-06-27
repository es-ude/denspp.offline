import unittest
from copy import deepcopy

import numpy as np

from denspp.offline.analog.adc import DefaultSettingsADC
from denspp.offline.analog.adc.adc_deltasigma import DeltaSigmaADC, SettingsADC


class TestDeltaSigmaADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(DefaultSettingsADC)

    def test_init(self):
        DeltaSigmaADC(settings_dev=self.sets, dac_order=1)

    def test_init_default_dac_order(self):
        method = DeltaSigmaADC(settings_dev=self.sets)
        self.assertEqual(method._DeltaSigmaADC__dac_order, 2)

    def _make_input(self, method: DeltaSigmaADC, num_samples: int = 200) -> np.ndarray:
        time = np.linspace(0, num_samples / self.sets.fs_ana, num_samples, endpoint=False)
        return self.sets.vcm + self.sets.dvref * np.sin(2 * np.pi * 10 * time)

    def test_adc_deltasigma_order_one_returns_ndarray(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_one(uin)
        self.assertIsInstance(result, np.ndarray)

    def test_adc_deltasigma_order_one_values_within_range(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_one(uin)
        max_val = 2 ** (self.sets.Nadc - 1) - 1 if self.sets.is_signed else 2**self.sets.Nadc - 1
        min_val = -(2 ** (self.sets.Nadc - 1)) if self.sets.is_signed else 0
        self.assertTrue(np.all(result <= max_val))
        self.assertTrue(np.all(result >= min_val))

    def test_adc_deltasigma_order_two_returns_ndarray(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_two(uin)
        self.assertIsInstance(result, np.ndarray)

    def test_adc_deltasigma_order_two_values_within_range(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_two(uin)
        max_val = 2 ** (self.sets.Nadc - 1) - 1 if self.sets.is_signed else 2**self.sets.Nadc - 1
        min_val = -(2 ** (self.sets.Nadc - 1)) if self.sets.is_signed else 0
        self.assertTrue(np.all(result <= max_val))
        self.assertTrue(np.all(result >= min_val))

    def test_adc_deltasigma_order_one_with_noise(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        method.use_noise = True
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_one(uin)
        self.assertIsInstance(result, np.ndarray)

    def test_adc_deltasigma_order_two_with_noise(self):
        method = DeltaSigmaADC(settings_dev=self.sets, dac_order=1)
        method.use_noise = True
        uin = self._make_input(method)
        result = method.adc_deltasigma_order_two(uin)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
