import numpy as np
from fractions import Fraction
from scipy.integrate import cumulative_trapezoid
from scipy.signal.windows import gaussian
from denspp.offline import check_key_elements, check_keylist_elements_all


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


def do_quantize_transient(transient_orig: dict, fs_new: float, u_lsb: float, i_gain: float=2e3) -> dict:
    """Performing a re-quantization of the transient input signal (amplitude and time)
    Args:
        transient_orig:     Input dictionary with transient signal ['V': voltage, 'I': current, 'fs': sampling rate]
        fs_new:             New sampling rate [Hz]
        u_lsb:              New smallest voltage resolution (least significant bit, LSB)
    Returns:
        Dictionary with new transient output ['V': voltage, 'I': current, 'fs': sampling rate]
    """
    assert check_keylist_elements_all(
        keylist=[key for key in transient_orig.keys()],
        elements=['V', 'I', 'fs']
    ), "Not all attributes of transiend_orig are included ['V', 'I', 'fs']"
    current0 = do_resample_time(transient_orig['I'], transient_orig['fs'], fs_new, do_offset_comp=True)
    voltage0 = do_resample_time(transient_orig['V'], transient_orig['fs'], fs_new)

    current_tran = do_resample_amplitude(i_gain * current0, u_lsb) / i_gain
    voltage_tran = do_resample_amplitude(voltage0, u_lsb)
    return {'I': current_tran, 'V': voltage_tran, 'fs': fs_new}


def do_resample_time(signal_in: np.ndarray, fs_orig: float, fs_new: float,
                     do_offset_comp: bool=False) -> np.ndarray:
    """Do resampling of time value from transient signals
    Args:
        signal_in:      Numpy array of transient input signal
        fs_orig:        Original sampling rate value
        fs_new:         New sampling rate value
        do_offset_comp: Do offset compensation on output
    Returns:
        Numpy array of resampled into
    """
    from scipy.signal import resample_poly

    u_chck = np.mean(signal_in)
    u_off = u_chck if not do_offset_comp else 0
    if not fs_orig == fs_new:
        p, q = Fraction(fs_new / fs_orig).limit_denominator(10000).as_integer_ratio()
        return u_off + resample_poly(signal_in - u_chck, p, q)
    else:
        return signal_in - u_off


def do_resample_amplitude(signal_in: np.ndarray, u_lsb: float) -> np.ndarray:
    """Do resampling of amplitude from transient signal
    Args:
        signal_in:  Numpy array with transient signal
        u_lsb:      New smallest voltage resolution (least significant bit, LSB)
    Returns:
        Numpy array with re-sampled input (amplitude)
    """
    return u_lsb * np.round(signal_in / u_lsb, 0)


def calculate_signal_integration(signal: np.ndarray, time: np.ndarray, initial: float=0.0) -> np.ndarray:
    """Calculating the injected charge amount of one stimulation pattern
    :param signal:      Numpy array with current input signal
    :param time:        Numpy array with timesamples [s]
    :param initial:     Floating value as initial charge value [C]
    :return:            Numpy array with injected charge amount during signal time
    """
    period = float(np.diff(time).mean())
    return cumulative_trapezoid(
        y=signal,
        x=time,
        dx=period,
        initial=initial,
        axis=-1
    )
