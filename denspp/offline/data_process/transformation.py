from typing import Optional

import numpy as np
from numpy import bartlett, blackman, hamming
from scipy.signal.windows import gaussian


def window_method(window_size: int, method: str = "hamming") -> np.ndarray:
    """Generating window for smoothing transformation method.
    :param window_size:     Integer number with size of the window
    :param method:          Selection of window method ['': None, 'Hamming', 'guassian', 'bartlett', 'blackman']
    :return:                Numpy array with window
    """
    methods_avai = {
        "hamming": hamming(window_size),
        "guassian": gaussian(window_size, int(0.16 * window_size), sym=True),
        "hanning": np.hanning(window_size),
        "bartlett": bartlett(window_size),
        "blackman": blackman(window_size),
    }

    window = np.ones(window_size)
    for key in methods_avai.keys():
        if method.lower() == key:
            window = methods_avai[key]
    return window


def do_fft(
    y: np.ndarray, fs: float, method_window: Optional[str] = None
) -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation.
    :param y:   Transient input signal
    :param fs:  Sampling rate [Hz]
    :param method_window:   Selected window
    :return:    Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    fft_in = y
    if method_window is not None:
        window = window_method(window_size=y.size, method=method_window)
        fft_in = window * y
    N = y.size // 2
    fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
    fft_out[0] = fft_out[0] / 2
    freq = fs * np.fft.fftfreq(fft_out.size)

    # Taking positive range
    xsel = np.where(freq >= 0)
    fft_out = fft_out[xsel]
    freq = freq[xsel]

    return freq, fft_out


def do_fft_withimag(
    y: np.ndarray, fs: float, method_window: Optional[str] = None
) -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation with imaginary part.

    :param y:   Transient input signal
    :param fs:  Sampling rate [Hz]
    :param method_window:   Selected window
    :return:    Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    fft_in = y
    if method_window is not None:
        window = window_method(window_size=y.size, method=method_window)
        fft_in = window * y
    fft_out = np.fft.rfft(fft_in)
    freq = np.fft.rfftfreq(fft_in.size, d=1 / fs)
    return freq, fft_out
