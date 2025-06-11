import numpy as np
from unittest import TestCase, main
from denspp.offline.analog.iv_polyfit import PolyfitIV


class TestPolynomFitting(TestCase):
    voltage_mea = np.linspace(start=-5.0, stop=+5.0, num=101, endpoint=True, dtype=float)
    current_mea = voltage_mea / 100e3

    def test_voltage_response(self):
        dut = PolyfitIV(
            sampling_rate=10e3,
            en_noise=False
        )
        dut.extract_polyfit_params(
            current=self.current_mea,
            voltage=self.voltage_mea
        )

        current_dut = dut.get_current(self.voltage_mea, 0.0)
        np.testing.assert_almost_equal(current_dut, self.current_mea, decimal=5)

    def test_current_response(self):
        dut = PolyfitIV(
            sampling_rate=10e3,
            en_noise=False
        )
        dut.extract_polyfit_params(
            current=self.current_mea,
            voltage=self.voltage_mea
        )

        voltage_dut = dut.get_voltage(self.current_mea)
        np.testing.assert_almost_equal(voltage_dut, self.voltage_mea, decimal=5)

    def test_get_params_single(self):
        dut = PolyfitIV(
            sampling_rate=10e3,
            en_noise=False
        )
        error = dut.extract_polyfit_params(
            current=self.current_mea,
            voltage=self.voltage_mea,
            find_best_order=False
        )
        chck = np.abs(error) < 1e-9
        self.assertTrue(chck)

    def test_get_params_optimize(self):
        dut = PolyfitIV(
            sampling_rate=10e3,
            en_noise=False
        )
        error = dut.extract_polyfit_params(
            current=self.current_mea,
            voltage=self.voltage_mea,
            find_best_order=True,
            order_range=[2, 10]
        )
        chck = np.abs(error) < 1e-9
        self.assertTrue(chck)


if __name__ == '__main__':
    main()
