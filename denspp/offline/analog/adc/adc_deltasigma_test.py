import unittest
from copy import deepcopy

from denspp.offline.analog.adc import DefaultSettingsADC
from denspp.offline.analog.adc.adc_deltasigma import DeltaSigmaADC, SettingsADC


class TestDeltaSigmaADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(DefaultSettingsADC)

    def test_init(self):
        DeltaSigmaADC(settings_dev=self.sets, dac_order=1)


if __name__ == "__main__":
    unittest.main()
