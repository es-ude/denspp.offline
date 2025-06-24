import numpy as np
from unittest import TestCase, main
from copy import deepcopy
from scipy.signal import find_peaks
from denspp.offline.data_process.transformation import do_fft
from denspp.offline.digital.dsp import SettingsFilter, DSP


test_settings = SettingsFilter(
    gain=1, fs=1e3,
    n_order=2, f_filt=[250],
    type='iir', f_type='butter', b_type='lowpass'
)


def extract_peaks(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq, trans = do_fft(y=signal, fs=fs)
    peakx, _ = find_peaks(x=trans, height=0.05)
    return freq[peakx], trans[peakx], peakx


class TestDigitalFilters(TestCase):
    time = np.linspace(start=0.0, stop=5.0, num=int(5.0* test_settings.fs), endpoint=False, dtype=float)
    freq = [10., 20., 50., 100., 200.]

    def test_signal_generation(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        f0, y0, _ = extract_peaks(signal, test_settings.fs)
        assert self.freq == f0.tolist()
        np.testing.assert_almost_equal(y0, [1.07953976, 1.07954007, 1.07954007, 1.07954005, 1.07954001], decimal=3)

    def test_lowpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'lowpass'
        sets.f_filt = [50.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.93, 0.71, 0.44, 0.21], decimal=1)

    def test_lowpass_iir_second_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 2
        sets.b_type = 'lowpass'
        sets.f_filt = [50.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.98, 0.71, 0.23, 0.05], decimal=1)

    def test_highpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'highpass'
        sets.f_filt = [50.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.19, .36, .71, 0.9, 0.98], decimal=1)

    def test_bandpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'bandpass'
        sets.f_filt = [50., 100.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.1, .21, .71, 0.71, 0.25], decimal=1)

    def test_bandstop_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'bandstop'
        sets.f_filt = [50., 100.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, .98, .71, 0.71, 0.97], decimal=1)

    def test_notch_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'notch'
        sets.f_filt = [50.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, .9, .001, 0.84, 0.97], decimal=1)

    def test_allpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'iir'
        sets.n_order = 1
        sets.b_type = 'allpass'
        sets.f_filt = [50.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, .99, .99, .99, 0.99], decimal=1)

    def test_lowpass_fir_taps21(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 21
        sets.b_type = 'lowpass'
        sets.f_filt = [50.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.92, 0.59, 0.09, 0.001], decimal=2)

    def test_lowpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 51
        sets.b_type = 'lowpass'
        sets.f_filt = [50.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.99, 0.5, 0.001, 0.001], decimal=2)

    def test_highpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 51
        sets.b_type = 'highpass'
        sets.f_filt = [20.0]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.26, 0.5, 0.99, 0.99, 0.99], decimal=2)

    def test_bandpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 51
        sets.b_type = 'bandpass'
        sets.f_filt = [20., 100.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.25, 0.49, 0.99, 0.49, 0.001], decimal=2)

    def test_bandstop_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 51
        sets.b_type = 'bandstop'
        sets.f_filt = [20., 100.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.88, 0.59, 0.01, 0.59, 1.18], decimal=2)

    def test_notch_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 501
        sets.b_type = 'notch'
        sets.f_filt = [50., 1.]
        result = DSP(sets).filter(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.99, 0.73, 0.99, 0.99], decimal=2)

    def test_allpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = 'fir'
        sets.n_order = 201
        sets.b_type = 'allpass'
        sets.f_filt = [0.]
        result = DSP(sets).filter(signal)
        np.testing.assert_almost_equal(signal[:-sets.n_order],result[sets.n_order:], decimal=5)


if __name__ == "__main__":
    main()
