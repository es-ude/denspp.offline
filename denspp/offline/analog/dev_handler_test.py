import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.dev_handler import generate_test_signal, SettingsDevice, ElectricalLoadHandler


TestSettings = SettingsDevice(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    params_use={'r': 100e3},
    temp=300,
    use_poly=False
)


class TestDeviceSettings(TestCase):
    t_sim_duration = 1e-3
    volt_peak = [2.0]
    volt_fsig = [2e3]
    volt_offset = 0.0
    test_time, test_volt = generate_test_signal(
        t_end=1.0e-3,
        fs=TestSettings.fs_ana,
        upp=volt_peak,
        fsig=volt_fsig,
        uoff=volt_offset
    )

    def test_temperature_voltage(self):
        set0 = deepcopy(TestSettings)
        temp_test = [100.0, 200.0, 250.0, 300.0, 350.0, 400.0]

        result = list()
        for value in temp_test:
            set0.temp = value
            result.append(set0.temperature_voltage)

        check = [0.008617333262145178, 0.017234666524290357, 0.021543333155362946, 0.025851999786435535, 0.030160666417508128, 0.034469333048580714]
        np.testing.assert_array_equal(np.asarray(result), np.asarray(check))

    def test_generate_signal_length_check(self):
        result = self.test_time.size
        check = self.test_volt.size
        np.testing.assert_equal(result, check)

    def test_generate_signal_time_length_value(self):
        result = self.test_time.size
        check = 1+int(TestSettings.fs_ana * self.t_sim_duration)
        np.testing.assert_equal(result, check)

    def test_generate_signal_time_length_delta(self):
        result = np.diff(self.test_time).mean()
        check = np.asarray(1/TestSettings.fs_ana)
        np.testing.assert_almost_equal(result, check, decimal=-6)

    def test_generate_signal_time_max(self):
        result = self.test_time.max()
        check = np.asarray(self.t_sim_duration)
        np.testing.assert_equal(result, check)

    def test_generate_signal_volt_max(self):
        result = self.test_volt.max()
        check = np.asarray(self.volt_peak[-1])
        np.testing.assert_almost_equal(result, check, decimal=-2)

    def test_generate_signal_volt_min(self):
        result = self.test_volt.min()
        check = np.asarray(self.volt_peak[-1])
        np.testing.assert_almost_equal(result, check, decimal=-2)

    def test_generate_signal_volt_mean(self):
        result = np.mean(self.test_volt)
        check = np.asarray(self.volt_offset)
        np.testing.assert_almost_equal(result, check, decimal=-9)

    def test_device_hndl_access(self):
        set0 = deepcopy(TestSettings)
        ovr_devices = ElectricalLoadHandler(settings_dev=set0).get_type_list()
        self.assertEqual(len(ovr_devices), 0)

    def test_device_get_voltage(self):
        set0 = deepcopy(TestSettings)
        try:
            volt = ElectricalLoadHandler(settings_dev=set0).get_voltage(self.test_volt / 1e6, 0.0)
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_device_get_current(self):
        set0 = deepcopy(TestSettings)
        try:
            curr = ElectricalLoadHandler(settings_dev=set0).get_current(self.test_volt, 0.0)
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_chck_violation_voltage_success(self):
        set0 = deepcopy(TestSettings)
        dut = ElectricalLoadHandler(settings_dev=set0)
        dut.change_boundary_voltage(self.test_volt.min() - 1, self.test_volt.max() + 1)
        rslt = dut.check_value_range_violation(self.test_volt, True)
        self.assertFalse(rslt)

    def test_chck_violation_voltage_failed(self):
        set0 = deepcopy(TestSettings)
        dut = ElectricalLoadHandler(settings_dev=set0)
        dut.change_boundary_voltage(self.test_volt.min() + 1, self.test_volt.max() - 1)
        rslt = dut.check_value_range_violation(self.test_volt, True)
        self.assertTrue(rslt)

    def test_chck_violation_current_success(self):
        set0 = deepcopy(TestSettings)
        dut = ElectricalLoadHandler(settings_dev=set0)
        dut.change_boundary_current(-10, -1)
        rslt = dut.check_value_range_violation(self.test_volt/1e3, False)
        self.assertFalse(rslt)

    def test_chck_violation_current_failed(self):
        set0 = deepcopy(TestSettings)
        dut = ElectricalLoadHandler(settings_dev=set0)
        dut.change_boundary_current(-10, -3)
        rslt = dut.check_value_range_violation(self.test_volt/1e3, False)
        self.assertTrue(rslt)

if __name__ == '__main__':
    main()
