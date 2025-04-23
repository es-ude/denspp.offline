from unittest import TestCase, main

import numpy as np
from denspp.offline.analog.common_func import CommonAnalogFunctions


class TestAnalogFunc(TestCase):
    method = CommonAnalogFunctions()
    input_clip = np.linspace(start=-3.0, stop=3.0, num=11, endpoint=True)

    def test_clip_type(self):
        output = self.method.clamp_voltage(self.input_clip)
        self.assertEqual(type(output), type(self.input_clip))

    def test_range_value(self):
        val = self.method.define_voltage_range(volt_low=-2.0, volt_hgh=2.0)
        self.assertEqual(val, [-2.0, 2.0])

    def test_vcm_bipolar(self):
        self.method.define_voltage_range(volt_low=-2.0, volt_hgh=2.0)
        self.assertEqual(self.method.vcm, 0.0)

    def test_vcm_unipolar(self):
        self.method.define_voltage_range(volt_low=0.0, volt_hgh=2.0)
        self.assertEqual(self.method.vcm, 1.0)

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


if __name__ == '__main__':
    main()