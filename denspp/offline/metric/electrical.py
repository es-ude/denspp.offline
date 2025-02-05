import numpy as np
from scipy.signal import find_peaks, correlate


def calculate_total_harmonics_distortion(freq: np.ndarray, spectral: np.ndarray,
                                         fsine: float, N_harmonics: int=4) -> [float, np.ndarray]:
    """Calculating the Total Harmonics Distortion (THD) of spectral input
    Args:
        freq:           Array with frequency values for spectral analysis
        spectral:       Array with Spectral input
        fsine:          Frequency of sinusoidal input from target
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB) and corresponding frequency positions of peaks
    """
    # --- Limiting the search space
    pos_x0 = np.argwhere(freq >= 0.5 * fsine).flatten()[0]
    pos_x1 = np.argwhere(freq >= (N_harmonics + 0.5) * fsine).flatten()[0]
    search_y = spectral[pos_x0:pos_x1]
    search_x = freq[pos_x0:pos_x1]

    # --- Getting peaks values
    df = np.mean(np.diff(freq))
    xpos, _ = find_peaks(search_y, distance=int(0.8 * fsine / df))
    peaks_y = search_y[xpos]

    # --- Getting THD and return
    total_harmonics = 20 * np.log10(np.sqrt(np.sum(np.power(peaks_y[1:], 2))) / peaks_y[0])
    return total_harmonics, search_x[xpos]


def calculate_cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray, do_print: bool=False) -> float:
    """Calculating the Cosine Similarity of two different inputs (same size)
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    out = correlate(y_pred / np.linalg.norm(y_pred), y_true / np.linalg.norm(y_true),'full', 'auto')
    cor_value = float(out[y_true.size])
    if do_print:
        print(f"\t Similarity coefficient = {100 * cor_value:.2f} %")
    return cor_value
