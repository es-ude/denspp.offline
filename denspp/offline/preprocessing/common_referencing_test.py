import unittest
import numpy as np
from copy import deepcopy
from .common_referencing import DefaultSettingsReferencing, CommonReferencing


class CommonReferencingTest(unittest.TestCase):
    set = deepcopy(DefaultSettingsReferencing)
    dut = CommonReferencing(set)

    num_samples = 1000
    signal_1d = 2*(np.random.rand(1, num_samples)-0.5)
    signal_2d = 2*(np.random.rand(100, num_samples)-0.5)
    signal_3d = 2*(np.random.rand(3,2, num_samples)-0.5)

    def test_dummy_build_1d(self):
        chck = np.array([True] * 1)
        rslt = self.dut.build_dummy_active_mapping(self.signal_1d)
        assert rslt.shape == (1, )
        np.testing.assert_array_equal(rslt, chck)

    def test_dummy_build_2d(self):
        chck = np.array([True] * 100)
        rslt = self.dut.build_dummy_active_mapping(self.signal_2d)
        assert rslt.shape == (100, )
        np.testing.assert_array_equal(rslt, chck)

    def test_dummy_build_3d(self):
        chck = np.array([[True] * 2] * 3)
        rslt = self.dut.build_dummy_active_mapping(self.signal_3d)
        assert rslt.shape == (3, 2)
        np.testing.assert_array_equal(rslt, chck)

    def test_reference_1d_one_channel_direct(self):
        chck = self.signal_1d.flatten()
        rslt = self.dut.get_reference_map(
            signal=self.signal_1d.flatten(),
            active=self.dut.build_dummy_active_mapping(self.signal_1d)
        )
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_almost_equal(rslt, chck, decimal=6)

    def test_reference_1d_one_channel(self):
        chck = self.signal_1d.flatten()
        rslt = self.dut.get_reference_map(
            signal=self.signal_1d,
            active=self.dut.build_dummy_active_mapping(self.signal_1d)
        )
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_array_equal(rslt, chck)

    def test_reference_1d_more_channels(self):
        chck = np.zeros(shape=(self.num_samples,))
        rslt = self.dut.get_reference_map(
            signal=self.signal_2d,
            active=self.dut.build_dummy_active_mapping(self.signal_2d)
        )
        assert rslt.shape == (self.num_samples,)
        self.assertLess(sum(rslt - chck) / self.num_samples, 0.01)

    def test_reference_2d_kernel_three(self):
        chck = np.zeros_like(self.signal_3d)
        rslt = self.dut.get_reference_map(
            signal=self.signal_3d,
            active=self.dut.build_dummy_active_mapping(self.signal_3d)
        )
        assert rslt.shape == (3, 2, self.num_samples)
        sum_rslt = float(np.sum(rslt-chck) / self.num_samples)
        self.assertLess(sum_rslt, 0.05)

    def test_car_1d_one_channel_direct(self):
        chck = np.zeros_like(self.signal_1d.flatten())
        rslt = self.dut.apply_reference(
            signal=self.signal_1d.flatten(),
            active=self.dut.build_dummy_active_mapping(self.signal_1d)
        )
        assert rslt.shape == (self.num_samples, )
        np.testing.assert_almost_equal(rslt, chck, decimal=6)

    def test_car_1d_one_channel(self):
        chck = self.signal_1d.flatten()
        rslt = self.dut.apply_reference(
            signal=self.signal_1d,
            active=self.dut.build_dummy_active_mapping(self.signal_1d)
        )
        assert rslt.shape == self.signal_1d.shape
        sum_rslt = np.sum(rslt-chck) / self.num_samples
        self.assertLess(sum_rslt, 0.03)

    def test_car_1d_more_channels(self):
        chck = np.zeros(shape=(self.num_samples,))
        rslt = self.dut.apply_reference(
            signal=self.signal_2d,
            active=self.dut.build_dummy_active_mapping(self.signal_2d)
        )
        assert rslt.shape == self.signal_2d.shape
        sum_rslt = np.sum(rslt - chck) / self.num_samples
        self.assertLess(sum_rslt, 0.01)

    def test_car_2d(self):
        chck = np.zeros(shape=(3, 2, self.num_samples))
        rslt = self.dut.apply_reference(
            signal=self.signal_3d,
            active=self.dut.build_dummy_active_mapping(self.signal_3d)
        )
        assert rslt.shape == self.signal_3d.shape
        sum_rslt = np.sum(rslt - chck) / self.num_samples
        self.assertLess(sum_rslt, 0.05)


if __name__ == '__main__':
    unittest.main()
