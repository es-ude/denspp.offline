import unittest
import numpy as np
from copy import deepcopy
from denspp.offline.metric.data_numpy import calculate_error_mae
from .cur_amp import SettingsCUR, DefaultSettingsCUR, CurrentAmplifier


class CurrentAmplifierTest(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsCUR = deepcopy(DefaultSettingsCUR)
        self.set0.offset_i = 0.0
        self.set0.offset_v = 0.0
        self.set0.fs_ana = 1e3

        self.dut = CurrentAmplifier(
            settings_dev=self.set0,
        )
        time = np.linspace(start=0., stop=1., num=int(1. * self.set0.fs_ana), endpoint=True)
        scale_amp = (self.set0.vdd - self.set0.vss) / 4
        self.current_in = scale_amp * np.sin(2*np.pi*time*5) / self.set0.transimpedance

    def test_settings_vcm_bipolar(self):
        set0: SettingsCUR = deepcopy(self.set0)
        set0.vdd = 0.9
        set0.vss = -0.9
        self.assertEqual(set0.vcm, 0.0)

    def test_settings_vcm_unipolar(self):
        set0: SettingsCUR = deepcopy(self.set0)
        set0.vdd = 0.9
        set0.vss = 0.0
        self.assertEqual(set0.vcm, 0.45)

    def test_instrumentation_amplifier(self):
        chck = self.current_in * self.set0.transimpedance + self.set0.vcm
        rslt = self.dut.instrumentation_amplifier(
            i_in=self.current_in,
            u_off=self.set0.vcm,
            v_gain=1.0
        )
        error = calculate_error_mae(rslt, chck)
        self.assertLess(error, 1e-5)

    def test_transimpedance_amplifier(self):
        chck = self.current_in * self.set0.transimpedance + self.set0.vcm
        rslt = self.dut.transimpedance_amplifier(
            i_in=self.current_in,
            u_ref=self.set0.vcm
        )
        error = calculate_error_mae(rslt, chck)
        self.assertLess(error, 1e-5)

    def test_push_amplifier(self):
        chck = self.current_in * self.set0.transimpedance + self.set0.vcm
        pos = np.argwhere(chck >= self.set0.vcm).flatten()
        chck[pos] = self.set0.vcm

        rslt = self.dut.push_amplifier(
            i_in=self.current_in,
            u_ref=self.set0.vcm
        )
        error = calculate_error_mae(rslt, chck)
        self.assertLess(error, 1e-5)

    def test_pull_amplifier(self):
        chck = self.current_in * self.set0.transimpedance + self.set0.vcm
        pos = np.argwhere(chck < self.set0.vcm).flatten()
        chck[pos] = self.set0.vcm

        rslt = self.dut.pull_amplifier(
            i_in=self.current_in,
            u_ref=self.set0.vcm
        )
        error = calculate_error_mae(rslt, chck)
        self.assertLess(error, 1e-5)

    def test_push_pull_amplifier(self):
        chck_pos = self.current_in * self.set0.transimpedance + self.set0.vcm
        chck_neg = deepcopy(chck_pos)
        pos = np.argwhere(chck_pos < self.set0.vcm).flatten()
        chck_pos[pos] = self.set0.vcm
        pos = np.argwhere(chck_neg >= self.set0.vcm).flatten()
        chck_neg[pos] = self.set0.vcm

        rslt = self.dut.push_pull_amplifier(
            i_in=self.current_in,
            u_ref=self.set0.vcm
        )
        error = calculate_error_mae(rslt[0], chck_pos)
        error += calculate_error_mae(rslt[1], chck_neg)
        self.assertLess(error, 1e-5)

    def test_push_pull_abs_amplifier(self):
        chck = np.abs(self.current_in * self.set0.transimpedance) + self.set0.vcm
        rslt = self.dut.push_pull_abs_amplifier(
            i_in=self.current_in,
            u_ref=self.set0.vcm
        )
        error = calculate_error_mae(rslt, chck)
        self.assertLess(error, 1e-5)


if __name__ == '__main__':
    unittest.main()
