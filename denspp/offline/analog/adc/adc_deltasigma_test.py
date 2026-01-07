import unittest
from copy import deepcopy
from .adc_deltasigma import SettingsADC, RecommendedSettingsADC, DeltaSigmaADC


class TestDeltaSigmaADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(RecommendedSettingsADC)

    def test_init(self):
        dut = DeltaSigmaADC(
            settings_dev=self.sets,
            dac_order=1
        )


if __name__ == '__main__':
    unittest.main()
