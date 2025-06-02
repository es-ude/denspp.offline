import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.dev_noise import SettingsNoise, ProcessNoise


test_settings = SettingsNoise(
    temp=300.0,
    wgn_dB=-120,
    Fc=10,
    slope=0.6
)


# --- Info: Function have to start with test_*
class TestSettingsNoise(TestCase):
    _sampling_rate: float = 20e3
    _stimuli: np.ndarray = np.zeros(shape=(1001, ), dtype=float)

    def test_temp_celsius_conversion_type(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        self.assertEqual(type(set0.temp_celsius), float)

    def test_temp_celsius_conversion_value(self):
        temp_test = [100.0, 200.0, 300.0]
        temp_ref = [-173.15, -73.15, 26.85]

        temp_result = list()
        set0 = deepcopy(test_settings)
        for temp in temp_test:
            set0.temp = temp
            temp_result.append(set0.temp_celsius)
        np.testing.assert_array_almost_equal(temp_result, temp_ref, decimal=10)

    def test_noise_pwr(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        self.assertAlmostEqual(set0.noise_pwr, 1.482e-21, delta=1e-24)

    def test_temp_voltage(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        self.assertAlmostEqual(set0.temperature_voltage, 25.85e-3, delta=5e-5)

    def test_handler_noise_awgn_dev_negative(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        set0.slope = 0.6
        set0.wgn_dB = -130
        set0.Fc = 10

        try:
            noise = ProcessNoise(settings=set0, fs_ana=self._sampling_rate).gen_noise_awgn_dev(
                size=self._stimuli.size,
                dev_e=-100e-9
            )
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_handler_noise_awgn_dev_positive(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        set0.slope = 0.6
        set0.wgn_dB = -130
        set0.Fc = 10

        noise = ProcessNoise(settings=set0, fs_ana=self._sampling_rate).gen_noise_awgn_dev(
            size=self._stimuli.size,
            dev_e=100e-9
        )
        noise_params = np.max(np.abs((noise.min(), noise.max())))
        self.assertTrue(noise_params < 5e-5)

    def test_handler_noise_awgn_dev_offset(self):
        set0 = deepcopy(test_settings)
        set0.temp = 300.0
        set0.slope = 0.6
        set0.wgn_dB = -130
        set0.Fc = 10

        noise = ProcessNoise(settings=set0, fs_ana=self._sampling_rate).gen_noise_awgn_dev(
            size=self._stimuli.size,
            dev_e=100e-9
        )
        ref = np.abs(np.mean(noise))
        noise_params = np.max(np.abs((noise.min(), noise.max())))
        # --- Should go to zero
        self.assertTrue(ref < 0.1 * noise_params)


if __name__ == '__main__':
    main()
