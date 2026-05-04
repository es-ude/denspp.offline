import unittest
from copy import deepcopy

from denspp.offline.analog.adc import DefaultSettingsADC
from denspp.offline.analog.adc.adc_flash import NyquistADC, SettingsADC


class TestFlashADC(unittest.TestCase):
    def setUp(self):
        self.sets: SettingsADC = deepcopy(DefaultSettingsADC)

    def test_init(self):
        NyquistADC(
            settings_dev=self.sets,
        )


if __name__ == "__main__":
    unittest.main()
