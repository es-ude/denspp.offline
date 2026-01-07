from unittest import TestCase, main
from copy import deepcopy
from denspp.offline.analog.adc.adc_settings import SettingsADC

TestSettingsADC = SettingsADC(
    vdd=+0.6,
    vss=-0.6,
    dvref=0.1,
    fs_ana=40e3,
    Nadc=12,
    fs_dig=20e3,
    osr=1,
    is_signed=False
)

class TestADCSettings(TestCase):
    def test_vcm_unipolar(self):
        set0 = deepcopy(TestSettingsADC)
        set0.vdd = +1.8
        set0.vss = 0.0
        self.assertEqual(set0.vcm, 0.9)

    def test_vcm_bipolar(self):
        set0 = deepcopy(TestSettingsADC)
        set0.vdd = +1.8
        set0.vss = -1.8
        self.assertEqual(set0.vcm, 0.0)

    def test_vref_unipolar(self):
        set0 = deepcopy(TestSettingsADC)
        set0.dvref = 0.1
        set0.vdd = +1.8
        set0.vss = 0.0
        self.assertAlmostEqual(set0.vref, (1.0, 0.8), delta=1e-4)

    def test_vref_bipolar(self):
        set0 = deepcopy(TestSettingsADC)
        set0.dvref = 0.1
        set0.vdd = +1.8
        set0.vss = -1.8
        self.assertAlmostEqual(set0.vref, (0.1, -0.1), delta=1e-4)

    def test_vref_range(self):
        set0 = deepcopy(TestSettingsADC)
        set0.dvref = 0.1
        set0.vdd = +1.8
        set0.vss = 0.0
        self.assertAlmostEqual(set0.vref_range, 0.2, delta=1e-4)

    def test_lsb(self):
        set0 = deepcopy(TestSettingsADC)
        set0.dvref = 0.1
        set0.Nadc = 2
        self.assertAlmostEqual(set0.lsb, 0.05, delta=1e-4)


if __name__ == '__main__':
    main()
