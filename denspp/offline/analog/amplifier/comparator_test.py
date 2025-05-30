import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.amplifier.comparator import SettingsComparator, Comparator


TestSettings = SettingsComparator(
    vdd=+1.8,
    vss=-1.8,
    offset=0.0,
    gain=1000,
    noise_dis=1e-12,
    hysteresis=0.25,
    out_analog=False,
    out_invert=False
)


class Test_AnalogComparator(TestCase):
    method = Comparator(settings_dev=TestSettings)
    data_tran_in = np.sin(2*np.pi * 1 * np.linspace(start=0, stop=2, num=21, endpoint=True))
    data_dc_in = method.vcm + np.array([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 0.5, 0.25, -0.25, -0.5, -1.0]) * TestSettings.vdd * TestSettings.hysteresis * 1.2


    def test_comp_vcm(self):
        result = Comparator(settings_dev=TestSettings).vcm
        self.assertEqual((TestSettings.vdd + TestSettings.vss) / 2, result)

    def test_comp_ideal_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False

        result = Comparator(settings_dev=set0).cmp_ideal(self.data_tran_in, self.method.vcm)
        ref = np.array([True, True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_ideal_voltage(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = True

        result = Comparator(settings_dev=set0).cmp_ideal(self.data_tran_in, self.method.vcm)
        ref = np.array([0.0000000e+00,  1.8000000e+00,  1.8000000e+00,  1.8000000e+00,  1.8000000e+00,  1.2246468e-13, -1.8000000e+00, -1.8000000e+00, -1.8000000e+00, -1.8000000e+00, -2.4492936e-13,  1.8000000e+00,  1.8000000e+00,  1.8000000e+00,  1.8000000e+00,  3.6739404e-13, -1.8000000e+00, -1.8000000e+00, -1.8000000e+00, -1.8000000e+00, -4.8985872e-13])
        np.testing.assert_array_almost_equal(result, ref, decimal=2)

    def test_comp_ideal_offset_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        set0.offset = 0.1

        data_in = self.method.vcm + set0.offset + np.array([-1, +1]) * 0.01
        result = Comparator(settings_dev=set0).cmp_ideal(data_in, self.method.vcm)
        ref = np.array([False, True])
        np.testing.assert_array_equal(result, ref)

    def test_comp_ideal_offset_boolean_inverted(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        set0.out_invert = True
        set0.offset = 0.1

        data_in = self.method.vcm + set0.offset + np.array([-1, +1]) * 0.01
        result = Comparator(settings_dev=set0).cmp_ideal(data_in, self.method.vcm)
        ref = np.array([True, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_ideal_normal_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        set0.offset = 0.1

        data_in = self.method.vcm + set0.offset + np.array([-1, +1]) * 0.01
        result = Comparator(settings_dev=set0).cmp_normal(data_in, self.method.vcm)
        ref = np.array([False, True])
        np.testing.assert_array_equal(result, ref)

    def test_comp_ideal_normal_boolean_inverted(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        set0.out_invert = True
        set0.offset = 0.1

        data_in = self.method.vcm + set0.offset + np.array([-1, +1]) * 0.01
        result = Comparator(settings_dev=set0).cmp_normal(data_in, self.method.vcm)
        ref = np.array([True, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_normal_positive_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        data_in = self.method.vcm + 0.125 * TestSettings.vdd

        result = Comparator(settings_dev=set0).cmp_normal(data_in, self.method.vcm)
        ref = np.array([True])
        np.testing.assert_array_equal(result, ref)

    def test_comp_normal_negative_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False
        data_in = self.method.vcm + 0.125 * TestSettings.vss

        result = Comparator(settings_dev=set0).cmp_normal(data_in, self.method.vcm)
        ref = np.array([False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_normal_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False

        result = Comparator(settings_dev=set0).cmp_normal(self.data_dc_in, self.method.vcm)
        ref = np.array([False, False, False, True, True, True, True, True, False, False, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_single_hysteresis_positive_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False

        result = Comparator(settings_dev=set0).cmp_single_pos_hysteresis(self.data_dc_in, self.method.vcm)
        ref = np.array([False, False, False, False, False, True, True, True, False, False, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_single_hysteresis_negative_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False

        result = Comparator(settings_dev=set0).cmp_single_neg_hysteresis(self.data_dc_in, self.method.vcm)
        ref = np.array([False, False, False, True, True, True, True, True, True, True, False])
        np.testing.assert_array_equal(result, ref)

    def test_comp_double_hysteresis_boolean(self):
        set0 = deepcopy(TestSettings)
        set0.out_analog = False

        result = Comparator(settings_dev=set0).cmp_double_hysteresis(self.data_dc_in, self.method.vcm)
        ref = np.array([False, False, False, False, False, True, True, True, True, True, False])
        np.testing.assert_array_equal(result, ref)


if __name__ == '__main__':
    main()
