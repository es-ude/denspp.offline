import numpy as np
from unittest import TestCase, main
from scipy.signal import find_peaks
from denspp.offline import is_close
from denspp.offline.data_process.transformation import transformation_window_method, do_fft


# --- Info: Function have to start with test_*
class TestTransformation(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2* np.pi * 100. * time) + 0.25 * np.sin(2* np.pi * 1000. * time)
    fs = float(1/np.diff(time).min())

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
        chck = freq.size == self.time.size / 2 and is_close(freq[-1], self.fs/2, 50.)
        self.assertTrue(chck)

    def test_fft_value(self):
        freq, spec = do_fft(
            y=self.vsig,
            fs=self.fs,
            method_window='hamming'
        )
        x = find_peaks(x=spec, wlen=2)
        chck_amp = all([is_close(spec[pred_frq], true_amp, 0.1) for pred_frq, true_amp in zip(x[0], [1.0, 0.25])])
        chck_frq = all([is_close(freq[pred_frq], true_frq, 0.1) for pred_frq, true_frq in zip(x[0], [100., 1000.])])
        self.assertTrue(chck_amp and chck_frq)


if __name__ == '__main__':
    main()
