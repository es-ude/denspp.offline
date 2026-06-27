import unittest
from copy import deepcopy

import numpy as np

from denspp.offline.analog.adc import DefaultSettingsADC
from denspp.offline.analog.adc.adc_flash import NyquistADC, SettingsADC


class TestFlashADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(DefaultSettingsADC)
        self.sets.fs_ana = self.sets.fs_dig

    def test_init(self):
        NyquistADC(
            settings_dev=self.sets,
        )

    def test_adc_nyquist_returns_tuple_of_ndarrays(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = self.sets.vcm + np.array([-0.5, -0.1, 0.0, 0.1, 0.5]) * self.sets.dvref
        x_out, quant_err = method.adc_nyquist(uin)
        self.assertIsInstance(x_out, np.ndarray)
        self.assertIsInstance(quant_err, np.ndarray)

    def test_adc_nyquist_output_size(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = self.sets.vcm + np.array([-0.5, -0.1, 0.0, 0.1, 0.5]) * self.sets.dvref
        x_out, quant_err = method.adc_nyquist(uin)
        self.assertEqual(x_out.size, uin.size)
        self.assertEqual(quant_err.size, uin.size)

    def test_adc_nyquist_within_digital_range(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = self.sets.vcm + np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * self.sets.dvref
        x_out, _ = method.adc_nyquist(uin)
        max_val = 2 ** (self.sets.Nadc - 1) - 1 if self.sets.is_signed else 2**self.sets.Nadc - 1
        min_val = -(2 ** (self.sets.Nadc - 1)) if self.sets.is_signed else 0
        self.assertTrue(np.all(x_out <= max_val))
        self.assertTrue(np.all(x_out >= min_val))

    def test_adc_nyquist_monotonic_for_increasing_input(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = self.sets.vcm + np.linspace(-0.5, 0.5, 20) * self.sets.dvref
        x_out, _ = method.adc_nyquist(uin)
        # Digitalwerte sollten mit steigender Eingangsspannung monoton nicht fallen
        diffs = np.diff(x_out)
        self.assertTrue(np.all(diffs >= 0))

    def test_adc_nyquist_single_sample(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = np.array([self.sets.vcm])
        x_out, quant_err = method.adc_nyquist(uin)
        self.assertEqual(x_out.size, 1)
        self.assertEqual(quant_err.size, 1)

    def test_adc_nyquist_at_lower_voltage_limit(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = np.array([self.sets.vref[1], self.sets.vref[1]])
        try:
            x_out, _ = method.adc_nyquist(uin)
        except IndexError:
            self.fail(
                "adc_nyquist raised IndexError at upper voltage limit - "
                "__partition_voltage likely doesn't cover vref[0]"
            )
        self.assertEqual(x_out.size, uin.size)
        check = np.array([0, 0])
        np.testing.assert_array_equal(x_out, check)

    def test_adc_nyquist_at_upper_voltage_limit(self):
        method = NyquistADC(settings_dev=self.sets)
        uin = np.array([self.sets.vref[0], self.sets.vref[0]])
        try:
            x_out, _ = method.adc_nyquist(uin)
        except IndexError:
            self.fail(
                "adc_nyquist raised IndexError at upper voltage limit - "
                "__partition_voltage likely doesn't cover vref[0]"
            )
        self.assertEqual(x_out.size, uin.size)
        check = np.array([4095, 4095])
        np.testing.assert_array_equal(x_out, check)


if __name__ == "__main__":
    unittest.main()
