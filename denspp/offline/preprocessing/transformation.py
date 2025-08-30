import numpy as np
from .window import transformation_window_method


def do_fft(y: np.ndarray, fs: float, method_window: str='hamming') -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation.
    :param y:               Transient input signal
    :param fs:              Sampling rate [Hz]
    :param method_window:   Selected window ['': None, 'Hamming', 'hanning', 'guassian', 'bartlett', 'blackman']
    :return:                Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    # Apply window method
    window = transformation_window_method(
        window_size=y.size,
        method=method_window
    )
    fft_in = window * y

    # Make transformation
    N = y.size // 2
    fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
    fft_out[0] = fft_out[0] / 2
    freq = fs * np.fft.fftfreq(fft_out.size)

    # Taking positive range
    xsel = np.where(freq >= 0.)
    fft_out = fft_out[xsel]
    freq = freq[xsel]
    return freq, fft_out


def do_fft_withimag(y: np.ndarray, fs: float, method_window: str='') -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation with imaginary part.
    :param y:   Transient input signal
    :param fs:  Sampling rate [Hz]
    :param method_window:   Selected window
    :return:    Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    window = transformation_window_method(
        window_size=y.size,
        method=method_window
    )
    fft_in = window * y

    fft_out = np.fft.rfft(fft_in)
    freq = np.fft.rfftfreq(fft_in.size, d=1 / fs)
    return freq, fft_out


def do_fft_inverse(y: np.ndarray, len_original: int) -> np.ndarray:
    """Perform inverse real FFT.
    :param y:               Fourier domain signal
    :param len_original:    Length of original time domain signal
    :return:                Time domain signal
    """
    return np.fft.irfft(y, n=len_original)
