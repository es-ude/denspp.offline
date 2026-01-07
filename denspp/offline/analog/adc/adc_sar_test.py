from unittest import TestCase, main
import numpy as np
from denspp.offline.analog.adc.adc_sar import SuccessiveApproximation
from denspp.offline.analog.adc.adc_settings import SettingsADC, SettingsNon


settings_adc = SettingsADC(
    vdd=1.8,
    vss=-1.8,
    dvref=900e-3,
    fs_ana=10e3,
    fs_dig=1e3,
    Nadc=12,
    osr=1,
    is_signed=True
)
RecommendedSettingsNon = SettingsNon(
    use_noise=False,
    wgndB=-100,
    offset=1e-6,
    gain_error=0.0
)


class TestSAR(TestCase):
    def setUp(self):
        self.method = SuccessiveApproximation(settings_adc)
        time = np.linspace(0, 1, int(settings_adc.fs_ana), endpoint=True, dtype=float)
        self.input = settings_adc.vcm + 0.2 * np.sin(2* np.pi* time* 100)


    def test_result_error(self):
        result = self.method.adc_sar(self.input)
        check = 1.2 * 2 * settings_adc.dvref / 2** settings_adc.Nadc
        self.assertEqual(result[2].max() <= check and np.abs(result[2].min()) <= check, True)

    def test_sar_sampling_rate(self):
        result = self.method.adc_sar(self.input)
        data_size = self.input.size / result[0].size
        samp_ratio = settings_adc.fs_ana / settings_adc.fs_dig
        self.assertEqual(data_size, samp_ratio)

    def test_result_type_digital(self):
        result = self.method.adc_sar(self.input)
        self.assertEqual(type(result[0]), np.ndarray)


if __name__ == '__main__':
    main()
