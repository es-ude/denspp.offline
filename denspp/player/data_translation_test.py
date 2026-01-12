import unittest
from unittest.mock import MagicMock
import numpy as np

from denspp.player.data_translation import DataTranslator 
from denspp.player.data_translation import BoardDataset
class TestDataTranslation(unittest.TestCase):
    def setUp(self):
        self.deplyoyed_data_translation = DataTranslator.__new__(DataTranslator)
        self.deplyoyed_data_translation._logger = MagicMock()
        self.deplyoyed_data_translation._data = BoardDataset(
            data=np.array([[-2,-1,0,1,2]]),
            samplingrate=1000,
            groundtruth=[],
            translation_value_voltage=None
        )
    

    def test_translate_data_for_oscilloscope_use_maxrange(self):
        self.deplyoyed_data_translation._device_name = "OscilloscopeMOX4"
        self.deplyoyed_data_translation._output_open = True

        self.deplyoyed_data_translation._translate_data_for_oscilloscope()
        np.testing.assert_array_almost_equal(self.deplyoyed_data_translation._data.data[0], [-10.,-5.,0.,5.,10.])


    def test_translate_data_for_oscilloscope_with_transaltion_value_voltage(self):
        self.deplyoyed_data_translation._device_name = "OscilloscopeMOX4"
        self.deplyoyed_data_translation._output_open = True
        self.deplyoyed_data_translation._data.translation_value_voltage = 0.5
    
        self.deplyoyed_data_translation._translate_data_for_oscilloscope()
        np.testing.assert_array_almost_equal(self.deplyoyed_data_translation._data.data[0], [-1.,-0.5,0.,0.5,1.])

if __name__ == '__main__':
    unittest.main()