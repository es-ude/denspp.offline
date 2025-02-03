from unittest import TestCase, main
import numpy as np
from denspp.offline.analog.adc.adc_sar import SuccessiveApproximation
from denspp.offline.analog.adc.adc_settings import SettingsADC, SettingsNon

settings_adc = SettingsADC(
    vdd=1.8,
    vss=-1.8,
    dvref=900e-3,
    fs_ana=100e3,
    fs_dig=10e3,
    Nadc=12,
    osr=1,
    type_out='signed'
)
RecommendedSettingsNon = SettingsNon(
    use_noise=False,
    wgndB=-100,
    offset=1e-6,
    gain_error=0.0
)


class TestSAR0(TestCase):
    method = SuccessiveApproximation(settings_adc)
    time = np.linspace(0, 1, int(settings_adc.fs_ana), endpoint=True, dtype=float)
    input = settings_adc.vcm + 0.2 * np.sin(2* np.pi* time* 100)
    result = method.adc_sar(input)

    def test_result_value(self):
        self.assertEqual(self.result, 10, "Should be 6")

    def test_result_type(self):
        self.assertEqual(type(self.result), type(int(0)), "Type should be integer")


if __name__ == '__main__':
    main()
