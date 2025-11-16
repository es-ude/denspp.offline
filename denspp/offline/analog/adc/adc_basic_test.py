from unittest import TestCase, main
import numpy as np
from denspp.offline.analog.adc.adc_basic import BasicADC
from denspp.offline.analog.adc.adc_settings import SettingsADC, SettingsNon


settings_adc = SettingsADC(
    vdd=1.8,
    vss=-1.8,
    dvref=750e-3,
    fs_ana=40e3,
    fs_dig=10e3,
    Nadc=12,
    osr=4,
    is_signed=True
)
RecommendedSettingsNon = SettingsNon(
    use_noise=False,
    wgndB=-100,
    offset=1e-6,
    gain_error=0.0
)


def inp_samp(time: np.ndarray) -> np.ndarray:
    freq = [4, 400]
    z = 0* time
    for f in freq:
        z += np.sin(2 * np.pi * f * time)
    return z / len(freq)


class TestBasic(TestCase):
    method = BasicADC(settings_adc)
    time = np.linspace(0, 1, int(settings_adc.fs_ana)+1, endpoint=True, dtype=float)

    input0 = settings_adc.vcm + np.array([-0.75, 0.5, -0.25, 0.01, +0.25, -0.5, +0.75])
    result0 = method.adc_ideal(input0)
    input1 = settings_adc.vcm + settings_adc.dvref * inp_samp(time)

    def test_adc(self):
        check = np.array([-2048, 1365, -683, 27, 682, -1366, 2047])
        np.testing.assert_array_equal(self.result0[0], check)

    def test_adc_size(self):
        check = self.input0.size
        self.assertEqual(self.result0[0].size, check)

    def test_adc_type(self):
        result = type(self.result0[0])
        self.assertEqual(result, np.ndarray)


if __name__ == '__main__':
    main()
