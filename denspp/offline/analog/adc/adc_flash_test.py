import unittest
from copy import deepcopy
from .adc_flash import SettingsADC, RecommendedSettingsADC, NyquistADC


class TestFlashADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(RecommendedSettingsADC)

    def test_init(self):
        dut = NyquistADC(
            settings_dev=self.sets,
        )


if __name__ == '__main__':
    unittest.main()
