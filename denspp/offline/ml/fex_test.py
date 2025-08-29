import unittest
import numpy as np
from copy import deepcopy
from .fex import SettingsFeature, DefaultSettingsFeature, FeatureExtraction


class FeatExtractionTest(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFeature = deepcopy(DefaultSettingsFeature)
        self.dut = FeatureExtraction(settings=self.set0)


if __name__ == '__main__':
    unittest.main()
