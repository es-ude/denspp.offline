import numpy as np
from unittest import TestCase, main
from scipy.signal import find_peaks
from denspp.offline.data_process.transformation import (transformation_window_method, do_fft,
                                                        do_resample_amplitude,
                                                        calculate_signal_integration)


def is_close(value: float, target: float, tolerance: float=0.05) -> bool:
    return abs(value - target) <= abs(target * tolerance)


# --- Info: Function have to start with test_*
class TestTransformation(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2* np.pi * 100. * time) + 0.25 * np.sin(2* np.pi * 1000. * time)
    fs = float(1/np.diff(time).min())

    def test_is_close_true(self):
        rslt = is_close(value=95, target=100, tolerance=0.05)
        self.assertTrue(rslt)

    def test_is_close_false(self):
        rslt = is_close(value=94, target=100, tolerance=0.05)
        self.assertFalse(rslt)

    def test_transformation_window_method_false(self):
        try:
            window = transformation_window_method(
                window_size=self.vsig.size,
                method = 'Hammingd'
            )
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_transformation_window_method_ones(self):
        window = transformation_window_method(
            window_size=self.vsig.size,
            method=''
        )
        np.testing.assert_array_equal(window, np.ones_like(self.vsig))

    def test_fft_spec(self):
        freq, spec = do_fft(
            y=self.vsig,
            fs=self.fs,
            method_window='hamming'
        )
        chck = freq.size == self.time.size / 2 and is_close(freq[-1], self.fs/2, 0.01)
        self.assertTrue(chck)

    def test_fft_value(self):
        freq, spec = do_fft(
            y=self.vsig,
            fs=self.fs,
            method_window='hamming'
        )
        x = find_peaks(x=spec, wlen=2)
        chck_amp = all([is_close(spec[pred_frq], true_amp, 0.1) for pred_frq, true_amp in zip(x[0], [1.0, 0.25])])
        chck_frq = all([is_close(freq[pred_frq], true_frq, 0.05) for pred_frq, true_frq in zip(x[0], [100., 1000.])])
        self.assertTrue(chck_amp and chck_frq)

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
