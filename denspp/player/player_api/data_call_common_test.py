import unittest
import data_call_common
import numpy as np
from unittest.mock import patch, Mock

class Call_Common_Test(unittest.TestCase):
    def setUp(self):
        self.data_controller = data_call_common.DataController()
        self.data_controller.logger = Mock()
        test_data_set(self)

# - - - -  Test for the "do_cut" methode - - - - 
    def test_do_cut_no_time_range(self):
        """Test do_cut when no time range is specified"""
        self.data_controller.t_range = []  # No time range specified
        self.data_controller.do_cut()
        # Expect data_raw to remain unchanged
        np.testing.assert_array_equal(self.data_controller.data_raw[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32))
    
    def test_do_cut_with_time_range(self):
        """Test do_cut when a valid time range is specified"""
        self.data_controller.t_range = [2, 4]
        self.data_controller.label_exist = False
        self.data_controller.do_cut()
        np.testing.assert_array_equal(self.data_controller.data_raw[0], np.array([2, 3], dtype=np.float32))
    
    def test_do_cut_with_labels(self):
        """Test do_cut when labels exist in the data controller"""
        self.data_controller.t_range = [2, 5]

        self.data_controller.do_cut()

        np.testing.assert_array_equal(self.data_controller.spike_xpos[0], np.array([0, 1]))
        np.testing.assert_array_equal(self.data_controller.cluster_id[0], np.array([2, 3]))


# - - - - Test for the "do_resample" methode - - - -
    def test_do_resample_no_change(self):
        """Test do_resample when the target sampling rate is the same as the current rate"""
        self.data_controller.fs_resample = 1  # Same as current fs
        original_data = self.data_controller.data_raw[0].copy()
        self.data_controller.do_resample()
        np.testing.assert_array_equal(self.data_controller.data_raw[0], original_data)
        self.data_controller.logger.warning.assert_called_with("Resampling skipped because the desired sampling rate is equal to the current sampling rate.")

    def test_do_resample_with_sampling(self):
        """Test do_resample when the target sampling rate is different"""
        pass
        self.data_controller.fs_resample = 2  # Different from current fs
        self.data_controller.data_raw = [np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0], dtype=np.float32)]
        self.data_controller.do_resample(num_points_mean=11)
        expected_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 2.5, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3, -2.5, -1.5, -1, -0.5, 0,], dtype=np.float32)
        np.testing.assert_array_almost_equal(self.data_controller.data_raw[0], expected_data)
        #TODO: Hinzufügen des Test; Umziehen auf die denspp.offline, somit kann test für call_common entfallen


def test_data_set(obj :object):
        """Test do_cut when labels exist in the data controller"""
        obj.data_controller.data_raw = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)]
        obj.data_controller.data_fs_orig = 1
        obj.data_controller.data_fs_current = 1
        obj.data_controller.label_exist = True
        obj.data_controller.spike_xpos = [np.array([0, 2, 3, 5, 6, 9])]
        obj.data_controller.cluster_id = [np.array([1, 2, 3, 4, 3, 2])]

if __name__ == '__main__':
    unittest.main()