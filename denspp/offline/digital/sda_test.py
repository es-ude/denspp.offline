import numpy as np
from unittest import TestCase, main
from copy import deepcopy
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA


TestSetting = SettingsSDA(
    fs=20e3,
    dx_sda=[1],
    mode_align=1,
    t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
    dt_offset=[0.1e-3, 0.1e-3],
    t_dly=0.3e-3,
    window_size=7,
    thr_gain=1.0,
    thr_min_value=100.0
)


class TestSettingsSDA(TestCase):
    def test_integer_negative(self):
        set0 = deepcopy(TestSetting)
        data_input = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset in data_input:
            set0.dt_offset[0] = offset
            result.append(set0.get_integer_for_negative_offset)

        check = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result, check)

    def test_integer_positive(self):
        set0 = deepcopy(TestSetting)
        data_input = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset in data_input:
            set0.dt_offset[1] = offset
            result.append(set0.get_integer_for_positive_offset)

        check = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result, check)

    def test_integer_total(self):
        set0 = deepcopy(TestSetting)
        data_input0 = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]
        data_input1 = [0.2e-3, 0.4e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.1e-3]

        result = list()
        for offset0, offset1 in zip(data_input0, data_input1):
            set0.dt_offset = [offset0, offset1]
            result.append(set0.get_integer_offset_total)

        check = [4, 10, 8, 12, 16, 12]
        np.testing.assert_array_equal(result, check)

    def test_integer_spike_size(self):
        set0 = deepcopy(TestSetting)
        data_input = [1.0e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3]

        result = list()
        for offset in data_input:
            set0.t_frame_lgth = offset
            result.append(set0.get_integer_spike_frame)

        check = [20, 24, 28, 32, 36]
        np.testing.assert_array_equal(result, check)

    def test_integer_spike_start(self):
        set0 = deepcopy(TestSetting)
        data_input = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset in data_input:
            set0.t_frame_start = offset
            result.append(set0.get_integer_spike_start)

        check = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result, check)


if __name__ == '__main__':
    main()
