import numpy as np
from unittest import TestCase, main
from denspp.offline.analog.func import do_resample_amplitude, calculate_signal_integration


def is_close(value: float, target: float, tolerance: float=0.05) -> bool:
    return abs(value - target) <= abs(target * tolerance)


# --- Info: Function have to start with test_*
class TestAnalogFunc(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2* np.pi * 100. * time) + 0.25 * np.sin(2* np.pi * 1000. * time)
    fs = float(1/np.diff(time).min())

    def test_is_close_true(self):
        rslt = is_close(value=95, target=100, tolerance=0.05)
        self.assertTrue(rslt)

    def test_is_close_false(self):
        rslt = is_close(value=94, target=100, tolerance=0.05)
        self.assertFalse(rslt)

    def test_resample_amplitude(self):
        signal = np.array([0.0, 0.1, 0.2, 0.05, -0.05, -0.15])
        rslt = do_resample_amplitude(
            signal_in=signal,
            u_lsb=0.1
        )
        chck = np.array([0.0, 0.1, 0.2, 0.0, -0.0, -0.1])
        np.testing.assert_array_equal(rslt, chck)


    def test_charge_calc_zero(self):
        qinj = calculate_signal_integration(
            signal=self.vsig,
            time=self.time
        )
        chck = qinj.size == self.vsig.size and abs(qinj[-1]) <= 1e-5
        self.assertTrue(chck)

    def test_charge_calc_sinus(self):
        qinj = calculate_signal_integration(
            signal=np.sin(2* np.pi* self.time * 10),
            time=self.time
        )
        chck = (1 - np.cos(2* np.pi* self.time * 10)) / (2* np.pi * 10)
        np.testing.assert_array_almost_equal(qinj, chck, decimal=4)


if __name__ == '__main__':
    main()
