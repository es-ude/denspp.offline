import numpy as np
from scipy.signal import find_peaks, correlate
from denspp.offline.preprocessing.transformation import do_fft


def calculate_total_harmonics_distortion(freq: np.ndarray, spectral: np.ndarray, N_harmonics: int=4) -> float:
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
    return 20 * np.log10(np.sqrt(np.sum(np.power(peaks_y[1:], 2))) / peaks_y[0])


def calculate_total_harmonics_distortion_from_transient(signal: np.ndarray, fs: float, N_harmonics: int=4) -> float:
    """Calculating the Total Harmonics Distortion (THD) from transient input
    Args:
        signal:         Array with frequency values for spectral analysis
        fs:             Sampling rate [Hz]
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB)
    """
    freq, spectral = do_fft(
        y=signal,
        fs=fs
    )
    return calculate_total_harmonics_distortion(
        freq=freq,
        spectral=spectral,
        N_harmonics=N_harmonics
    )


def calculate_cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculating the Cosine Similarity of two different inputs (same size)
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    out = correlate(y_pred / np.linalg.norm(y_pred), y_true / np.linalg.norm(y_true),'full', 'auto')
    return float(out[y_true.size-1])
