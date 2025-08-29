from unittest import TestCase, main
from copy import deepcopy
import numpy as np
from denspp.offline.analog.common_func import CommonAnalogFunctions, CommonDigitalFunctions


class TestAnalogFunc(TestCase):
    method = CommonAnalogFunctions()
    input_clip = np.linspace(start=-3.0, stop=3.0, num=11, endpoint=True)

    def test_clip_type(self):
        output = self.method.clamp_voltage(self.input_clip)
        self.assertEqual(type(output), type(self.input_clip))

    def test_range_value(self):
        val = self.method.define_voltage_range(volt_low=-2.0, volt_hgh=2.0)
        self.assertEqual(val, [-2.0, 2.0])

    def test_clamp_numpy_value(self):
        self.method.define_voltage_range(volt_low=-2.0, volt_hgh=1.0)
        output = self.method.clamp_voltage(self.input_clip)
        ref = np.array([-2., -2., -1.8, -1.2, -0.6,  0,  0.6,  1.,  1.,  1., 1.])
        self.assertLess(np.sum(np.abs(output-ref)), 1e-12)

    def test_clamp_numpy_type(self):
        self.method.define_voltage_range(volt_low=-2.0, volt_hgh=1.0)
        output = self.method.clamp_voltage(self.input_clip)
        self.assertEqual(type(output), type(self.input_clip))

    def test_clamp_float_value(self):
        self.method.define_voltage_range(volt_low=-2.0, volt_hgh=1.0)
        output = self.method.clamp_voltage(1.45)
        ref = 1.0
        self.assertLess(np.sum(np.abs(output-ref)), 1e-12)

    def test_clamp_float_type(self):
        self.method.define_voltage_range(volt_low=-2.0, volt_hgh=1.0)
        output = self.method.clamp_voltage(1.45)
        self.assertEqual(type(output), type(1.0))


class TestDigitalFunc(TestCase):
    method = CommonDigitalFunctions()
    input_clip = np.linspace(start=-3.0, stop=3.0, num=11, endpoint=True)
    input_quant = np.random.random(size=(10,2))

    def test_range_unsigned_2_0(self):
        range_val = self.method.define_limits(bit_signed=False, total_bitwidth=2, frac_bitwidth=0)
        ref = np.array([0.0, 3.0])
        np.testing.assert_array_equal(range_val, ref)

    def test_range_signed_2_0(self):
        range_val = self.method.define_limits(bit_signed=True, total_bitwidth=2, frac_bitwidth=0)
        ref = np.array([-2.0, 1.0])
        np.testing.assert_array_equal(range_val, ref)

    def test_range_unsigned_6_2(self):
        range_val = self.method.define_limits(bit_signed=False, total_bitwidth=6, frac_bitwidth=2)
        ref = np.array([0.0, 15.75])
        np.testing.assert_array_equal(range_val, ref)

    def test_range_signed_6_2(self):
        range_val = self.method.define_limits(bit_signed=True, total_bitwidth=6, frac_bitwidth=2)
        ref = np.array([-8.0, 7.75])
        np.testing.assert_array_equal(range_val, ref)

    def test_clip_type(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=2, frac_bitwidth=0)
        output = self.method.clamp_digital(self.input_clip)
        self.assertEqual(type(output), type(self.input_clip))

    def test_clip_signed_2_0(self):
        self.method.define_limits(bit_signed=True, total_bitwidth=2, frac_bitwidth=0)
        output = self.method.clamp_digital(self.input_clip)
        ref = np.array([-2., -2., -1.8, -1.2, -0.6,  0,  0.6,  1.,  1.,  1., 1.])
        self.assertLess(np.sum(np.abs(output-ref)), 1e-12)

    def test_clip_unsigned_1_0(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=1, frac_bitwidth=0)
        output = self.method.clamp_digital(self.input_clip)
        ref = np.array([0., 0., 0., 0., 0., 0., 0.6, 1., 1., 1., 1.])
        self.assertLess(np.sum(np.abs(output-ref)), 1e-12)

    def test_clip_signed_3_2(self):
        self.method.define_limits(bit_signed=True, total_bitwidth=3, frac_bitwidth=2)
        output = self.method.clamp_digital(self.input_clip)
        ref = np.array([-1., -1., -1., -1., -0.6, 0., 0.6, .75, 0.75, 0.75, 0.75])
        self.assertLess(np.sum(np.abs(output-ref)), 1e-12)

    def test_clip_unsigned_3_2(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=4, frac_bitwidth=2)
        output = self.method.clamp_digital(self.input_clip)
        ref = np.array([0., 0., 0., 0., 0., 0., 0.6, 1.2, 1.8, 2.4, 3.])
        self.assertLess(np.sum(np.abs(output - ref)), 1e-12)

    def test_quantize_type_float(self):
        input = 1.2
        self.method.define_limits(bit_signed=False, total_bitwidth=2, frac_bitwidth=0)
        output = self.method.quantize_fxp(input)
        self.assertEqual(type(output), type(input))

    def test_quantize_type_numpy(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=2, frac_bitwidth=0)
        output = self.method.quantize_fxp(self.input_quant)
        self.assertEqual(type(output), type(self.input_quant))

    def test_quantize_signed_4_2(self):
        self.method.define_limits(bit_signed=True, total_bitwidth=4, frac_bitwidth=2)
        output = self.method.quantize_fxp(self.input_clip)
        ref = np.array([-2., -2., -1.75, -1.25, -0.5, 0., 0.5, 1.25, 1.75, 1.75, 1.75])
        chck = np.sum(np.abs(output - ref), axis=0)
        self.assertLess(chck, 1e-12)

    def test_quantize_signed_4_2_max(self):
        self.method.define_limits(bit_signed=True, total_bitwidth=4, frac_bitwidth=2)
        input = 4 * (self.input_quant - 0.5)
        output = self.method.quantize_fxp(input)
        chck = np.max(np.abs(input - output))
        self.assertLess(chck, 2**-2)

    def test_quantize_unsigned_4_2(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=4, frac_bitwidth=2)
        output = self.method.quantize_fxp(self.input_clip)
        ref = np.array([0., 0., 0., 0., 0., 0., 0.5, 1.25, 1.75, 2.5, 3.])
        chck = np.sum(np.abs(output - ref), axis=0)
        self.assertLess(chck, 1e-12)

    def test_quantize_unsigned_4_2_max(self):
        self.method.define_limits(bit_signed=False, total_bitwidth=4, frac_bitwidth=2)
        input = 4 * (self.input_quant - 0.0)
        output = self.method.quantize_fxp(input)
        chck = np.max(np.abs(input - output))
        self.assertLess(chck, 2**-2)

    def test_extract_rising_edge_single(self):
        stimulus = np.array([False, False, False, True, True, True, True, True, False, False], dtype=bool)
        points = self.method.extract_rising_edge(stimulus)
        self.assertEqual(points, [3])

    def test_extract_rising_edge_double(self):
        stimulus = np.array([False, False, True, True, False, False, True, True, False], dtype=bool)
        points = self.method.extract_rising_edge(stimulus)
        self.assertEqual(points, [2, 6])

    def test_extract_falling_edge_single(self):
        stimulus = np.array([False, False, False, True, True, True, True, True, False, False], dtype=bool)
        points = self.method.extract_falling_edge(stimulus)
        self.assertEqual(points, [8])

    def test_extract_falling_edge_double(self):
        stimulus = np.array([False, False, True, True, False, False, True, True, False], dtype=bool)
        points = self.method.extract_falling_edge(stimulus)
        self.assertEqual(points, [4, 8])


if __name__ == '__main__':
    main()