import numpy as np
from scipy.signal.windows import gaussian
from denspp.offline import check_key_elements


def transformation_window_method(window_size: int, method: str= "hamming") -> np.ndarray:
    """Generating window for smoothing input of signal transformation method.
    :param window_size:     Integer number with size of the window
    :param method:          Selection of window method ['': Ones, 'hamming', 'hanning', 'guassian', 'bartlett', 'blackman']
    :return:                Numpy array with window
    """
    methods_avai = {
        '': np.ones(window_size),
        "hamming": np.hamming(window_size),
        "guassian": gaussian(window_size, int(0.16 * window_size), sym=True),
        "hanning": np.hanning(window_size),
        "bartlett": np.bartlett(window_size),
        "blackman": np.blackman(window_size),
    }
    methods_check = [method.lower() for method in methods_avai.keys()]
    assert check_key_elements(method.lower(), methods_check), f"Wrong method ({methods_check})"
    return methods_avai[[key for key in methods_check if key == method.lower()][0]]


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
