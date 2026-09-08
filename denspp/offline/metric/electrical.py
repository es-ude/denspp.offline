import numpy as np
from scipy.signal import find_peaks

from denspp.offline.preprocessing import do_fft


def calculate_total_harmonics_distortion(
    freq: np.ndarray, spectral: np.ndarray, N_harmonics: int = 4
) -> float:
    """Calculating the Total Harmonics Distortion (THD) of spectral input
    Args:
        freq:           Array with frequency values for spectral analysis
        spectral:       Array with Spectral input
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB) and corresponding frequency positions of peaks
    """
    fsine = freq[np.argmax(spectral).flatten()[0]]
    # --- Limiting the search space
    pos_x0 = np.argwhere(freq >= 0.5 * fsine).flatten()[0]
    pos_x1 = np.argwhere(freq >= (N_harmonics + 1.5) * fsine).flatten()[0]
    search_y = spectral[pos_x0:pos_x1]

    # --- Getting peaks values
    df = np.mean(np.diff(freq))
    xpos, _ = find_peaks(search_y, distance=int(0.8 * fsine / df))
    peaks_y = search_y[xpos]

    # --- Return THD
    return float(20 * np.log10(np.sqrt(np.sum(np.power(peaks_y[1:], 2))) / peaks_y[0]))


def calculate_total_harmonics_distortion_from_transient(
    signal: np.ndarray, fs: float, N_harmonics: int = 4
) -> float:
    """Calculating the Total Harmonics Distortion (THD) from transient input
    Args:
        signal:         Array with frequency values for spectral analysis
        fs:             Sampling rate [Hz]
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB)
    """
    freq, spectral = do_fft(y=signal, fs=fs)
    return calculate_total_harmonics_distortion(freq=freq, spectral=spectral, N_harmonics=N_harmonics)


def calculate_cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculating the Cosine Similarity of two different inputs (same size)
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    norm_pred = np.linalg.norm(y_pred)
    norm_true = np.linalg.norm(y_true)
    if norm_pred == 0 or norm_true == 0:
        raise ValueError("Cosine Similarity is not defined for NULL vectors")
    cos_sim = np.dot(y_pred.ravel(), y_true.ravel()) / (norm_pred * norm_true)
    return float(np.clip(cos_sim, -1.0, 1.0))


def calculate_sinad_from_spectrum(
    spectrum: np.ndarray, num_harmonics: int = 1, leakage_bins: int = 3
) -> float:
    """Calculating the Signal-to-Noise-and-Distortion Ratio (SINAD) of a signal from spectrum input
    :param spectrum:        Numpy array with signal (shape = (num_samples, ))
    :param num_harmonics:   Number of used harmonics
    :return:                SINAD in dB
    """
    power = np.abs(spectrum) ** 2
    fundamental_idx = np.argmax(power)

    signal_power = 0.0
    for ite in range(num_harmonics):
        lo = max(0, (ite + 1) * fundamental_idx - leakage_bins)
        hi = min(len(power), (ite + 1) * fundamental_idx + leakage_bins + 1)
        signal_power += np.sum(power[lo:hi])

    noise_and_distortion_power = np.sum(power) - signal_power
    return float(10 * np.log10(signal_power / noise_and_distortion_power))


def calculate_sinad_from_transient(signal: np.ndarray, fs: float, num_harmonics: int = 1) -> float:
    """Calculating the Signal-to-Noise-and-Distortion Ratio (SINAD) of a signal from transient signal input
    :param signal:          Numpy array with signal (shape = (num_samples, ))
    :param fs:              Sampling rate [Hz]
    :param num_harmonics:   Number of used harmonics
    :return:                SINAD in dB
    """
    spectrum = do_fft(
        y=signal,
        fs=fs,
        method_window="hamming",
    )[1]
    return calculate_sinad_from_spectrum(spectrum=spectrum, num_harmonics=num_harmonics)


def calculate_enob_from_spectrum(spectrum: np.ndarray, num_harmonics: int = 1) -> float:
    """Calculating the effective number of bits (ENOB) from spectrum
    :param spectrum:        Numpy array with spectrum (shape = (num_samples, ))
    :param num_harmonics:   Number of used harmonics
    :return:                ENOB in dB
    """
    sinad_db = calculate_sinad_from_spectrum(spectrum=spectrum, num_harmonics=num_harmonics)
    return (sinad_db - 1.76) / 6.02


def calculate_enob_from_transient(signal: np.ndarray, fs: float, num_harmonics: int = 1) -> float:
    """Calculating the effective number of bits (ENOB) from spectrum
    :param signal:          Numpy array with transient input (shape = (num_samples, ))
    :param fs:              Sampling rate [Hz]
    :param num_harmonics:   Number of used harmonics
    :return:                ENOB in dB
    """
    spectrum = do_fft(
        y=signal,
        fs=fs,
        method_window="hamming",
    )[1]
    return calculate_enob_from_spectrum(spectrum=spectrum, num_harmonics=num_harmonics)
