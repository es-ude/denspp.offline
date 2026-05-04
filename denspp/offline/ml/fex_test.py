import unittest
from copy import deepcopy

from .fex import DefaultSettingsFeature, FeatureExtraction, SettingsFeature


class FeatExtractionTest(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFeature = deepcopy(DefaultSettingsFeature)
        self.dut = FeatureExtraction(settings=self.set0)


if __name__ == "__main__":
    unittest.main()
