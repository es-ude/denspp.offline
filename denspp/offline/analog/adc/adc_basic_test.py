from unittest import TestCase, main

import numpy as np

from denspp.offline.analog.adc.adc_basic import BasicADC
from denspp.offline.analog.adc.adc_settings import SettingsADC, SettingsNon

settings_adc = SettingsADC(
    vdd=1.8,
    vss=-1.8,
    dvref=750e-3,
    fs_ana=40e3,
    fs_dig=10e3,
    Nadc=12,
    osr=4,
    is_signed=True,
)
RecommendedSettingsNon = SettingsNon(use_noise=False, wgndB=-100, offset=1e-6, gain_error=0.0)


def inp_samp(time: np.ndarray) -> np.ndarray:
    freq = [4, 400]
    z = 0 * time
    for f in freq:
        z += np.sin(2 * np.pi * f * time)
    return z / len(freq)


class TestBasic(TestCase):
    method = BasicADC(settings_adc)
    time = np.linspace(0, 1, int(settings_adc.fs_ana) + 1, endpoint=True, dtype=float)

    input0 = settings_adc.vcm + np.array([-0.75, 0.5, -0.25, 0.01, +0.25, -0.5, +0.75])
    result0 = method.adc_ideal(input0)
    input1 = settings_adc.vcm + settings_adc.dvref * inp_samp(time)

    def test_adc(self):
        check = np.array([-2048, 1365, -683, 27, 682, -1366, 2047])
        np.testing.assert_array_equal(self.result0[0], check)

    def test_adc_size(self):
        check = self.input0.size
        self.assertEqual(self.result0[0].size, check)

    def test_adc_type(self):
        result = type(self.result0[0])
        self.assertEqual(result, np.ndarray)

    def test_snr_ideal(self):
        check = 10 * np.log10(4) * settings_adc.Nadc + 10 * np.log10(3 / 2)
        self.assertAlmostEqual(self.method.snr_ideal, check)

    def test_snr_ideal_type(self):
        self.assertIsInstance(self.method.snr_ideal, float)

    def test_do_snh_stream_size(self):
        result = self.method.do_snh_stream(self.input1, f_snh=1e3)
        self.assertEqual(result.size, self.input1.size)

    def test_do_snh_stream_type(self):
        result = self.method.do_snh_stream(self.input1, f_snh=1e3)
        self.assertIsInstance(result, np.ndarray)

    def test_adc_ideal_single(self):
        input = settings_adc.vcm + np.array([-0.75])
        check = np.array([[-2048], [-1.5], [7.499923e-01]])
        rslt = self.method.adc_ideal(input)
        np.testing.assert_almost_equal(rslt, check, decimal=5)

    def test_do_snh_stream_holds_value(self):
        result = self.method.do_snh_stream(self.input1, f_snh=1e3)
        self.assertFalse(np.array_equal(result, self.input1))

    def test_generate_sar_empty_data_shapes(self):
        x_out, u_out, u_err = BasicADC._generate_sar_empty_data(shape=(5,))
        self.assertEqual(x_out.shape, (5,))
        self.assertEqual(u_out.shape, (5,))
        self.assertEqual(u_err.shape, (5,))

    def test_generate_sar_empty_data_dtypes(self):
        x_out, u_out, u_err = BasicADC._generate_sar_empty_data(shape=(5,))
        self.assertEqual(x_out.dtype, np.int16)
        self.assertEqual(u_out.dtype, np.float32)
        self.assertEqual(u_err.dtype, np.float32)

    def test_generate_sar_empty_data_zeros(self):
        x_out, u_out, u_err = BasicADC._generate_sar_empty_data(shape=(5,))
        np.testing.assert_array_equal(x_out, np.zeros(5, dtype=np.int16))
        np.testing.assert_array_equal(u_out, np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(u_err, np.zeros(5, dtype=np.float32))

    def test_generate_dsigma_empty_data_shapes(self):
        x_out_hs, x_bit = BasicADC._generate_dsigma_empty_data(shape=(5,))
        self.assertEqual(x_out_hs.shape, (5,))
        self.assertEqual(x_bit.shape, (5,))

    def test_generate_dsigma_empty_data_dtypes(self):
        x_out_hs, x_bit = BasicADC._generate_dsigma_empty_data(shape=(5,))
        self.assertEqual(x_out_hs.dtype, np.int32)
        self.assertEqual(x_bit.dtype, np.int32)

    def test_generate_dsigma_empty_data_zeros(self):
        x_out_hs, x_bit = BasicADC._generate_dsigma_empty_data(shape=(5,))
        np.testing.assert_array_equal(x_out_hs, np.zeros(5, dtype=np.int32))
        np.testing.assert_array_equal(x_bit, np.zeros(5, dtype=np.int32))


if __name__ == "__main__":
    main()
