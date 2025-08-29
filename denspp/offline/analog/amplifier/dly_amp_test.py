import unittest
import numpy as np
from copy import deepcopy
from denspp.offline.metric.data_numpy import calculate_error_mae
from .dly_amp import SettingsDLY, DefaultSettingsDLY, DelayAmplifier


class DelayAmplifierTest(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsDLY = deepcopy(DefaultSettingsDLY)
        self.set0.t_dly = 10e-3
        self.set0.fs_ana = 1e3

        self.dut = DelayAmplifier(
            settings_dev=self.set0,
        )
        time = np.linspace(start=0., stop=1., num=int(1. * self.set0.fs_ana), endpoint=True)
        scale_amp = (self.set0.vdd - self.set0.vss) / 4
        self.signal_in = scale_amp * np.sin(2*np.pi*time*5)

    def test_settings_vcm_bipolar(self):
        set0: SettingsDLY = deepcopy(self.set0)
        set0.vdd = 0.9
        set0.vss = -0.9
        self.assertEqual(set0.vcm, 0.0)

    def test_settings_vcm_unipolar(self):
        set0: SettingsDLY = deepcopy(self.set0)
        set0.vdd = 0.9
        set0.vss = 0.0
        self.assertEqual(set0.vcm, 0.45)

    def test_simple_delay(self):
        chck = self.signal_in + self.set0.vcm
        rslt = self.dut.do_simple_delay(
            u_inp=self.signal_in
        )
        error = calculate_error_mae(rslt[self.set0.num_dly_taps:], chck[:-self.set0.num_dly_taps])
        self.assertLess(error, 1e-5)

    def test_recursive_delay(self):
        chck = self.signal_in + self.set0.vcm
        rslt = self.dut.do_recursive_delay(
            u_inp=self.signal_in
        )
        error = calculate_error_mae(rslt[self.set0.num_dly_taps:], chck[:-self.set0.num_dly_taps])
        self.assertLess(error, 1e-5)

    def test_allpass_first_order_delay(self):
        chck = self.signal_in + self.set0.vcm
        rslt = self.dut.do_allpass_first_order(
            u_in=self.signal_in
        )
        steps = int(self.set0.num_dly_taps / np.pi)
        error = calculate_error_mae(rslt[steps:], chck[:-steps])
        self.assertLess(error, 1e-3)

    def test_allpass_second_order_delay(self):
        chck = self.signal_in + self.set0.vcm
        rslt = self.dut.do_allpass_second_order(
            u_in=self.signal_in,
            bandwidth=100.
        )
        steps = int(self.set0.num_dly_taps/np.pi)
        error = calculate_error_mae(rslt[steps:], chck[:-steps])
        self.assertLess(error, 1e-2)


if __name__ == '__main__':
    unittest.main()
