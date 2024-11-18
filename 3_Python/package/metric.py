import numpy as np
from scipy.signal import find_peaks, correlate


def calculate_snr(yin: np.ndarray, ymean: np.ndarray) -> np.ndarray:
    """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform"""
    a0 = (np.max(ymean) - np.min(ymean)) ** 2
    b0 = np.sum((yin - ymean) ** 2)
    return 10 * np.log10(a0 / b0)


def calculate_error_mbe(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with mean bias error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(y_pred - y_true) / y_pred.size)
    else:
        error = y_pred - y_true
    if do_print:
        print(f"Mean bias error (MBE) = {error:.4f}")
    return error


def calculate_error_mae(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with mean absolute error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(np.abs(y_pred - y_true)) / y_pred.size)
    else:
        error = float(np.abs(y_pred - y_true))
    if do_print:
        print(f"Mean absolute error (MAE) = {error:.4f}")
    return error


def calculate_error_mse(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with mean squared error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        error = float(np.sum((y_pred - y_true) ** 2) / y_pred.size)
    else:
        error = float(y_pred - y_true) ** 2
    if do_print:
        print(f"Mean squared error (MSE) = {error:.4f}")
    return error


def calculate_error_mpe(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with mean percentage error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        error = float(np.sum((y_true - y_pred) / y_true) / y_true.size)
    else:
        error = float((y_true - y_pred) / y_true)
    if do_print:
        print(f"Mean percentage error (MPE) = {error:.4f}")
    return error


def calculate_error_mape(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with mean absolute percentage error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(np.abs(y_true - y_pred) / np.abs(y_true)) / y_true.size)
    else:
        error = float(abs(y_true - y_pred) / abs(y_true))
    if do_print:
        print(f"Mean absolute percentage error (MAPE) = {error:.4f}")
    return error


def calculate_error_rae(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with relative absolute error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    mse = calculate_error_mse(y_pred, y_true)
    if isinstance(y_pred, np.ndarray):
        error = float(np.sqrt(mse) / calculate_error_mae(np.zeros(shape=y_pred.shape), y_true))
    else:
        error = float((mse ** 0.5) / calculate_error_mae(0.0, y_pred))
    if do_print:
        print(f"Relative absolute error (RAE) = {error:.4f}")
    return error


def calculate_error_rse(y_pred: np.ndarray | float, y_true: np.ndarray | float, do_print=False) -> float:
    """Calculating the distance-based metric with relative squared error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    mse = calculate_error_mse(y_pred, y_true)
    y_true_mean = np.mean(y_true)
    if isinstance(y_pred, np.ndarray):
        error = float(mse / np.sum((y_pred - y_true_mean) ** 2) / y_pred.size)
    else:
        error = float(mse / (y_pred - y_true_mean) ** 2)
    if do_print:
        print(f"Relative squared error (RSE) = {error:.4f}")
    return error


def calculate_error_rmse(y_pred: np.ndarray, y_true: np.ndarray, do_print=False) -> float:
    """Calculating the Root Mean Squared Error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    error = np.sqrt(np.sum(np.abs(y_pred - y_true) ** 2) / y_pred.size)
    if do_print:
        print(f"Root mean squared error (MSE) = {error:.4f}")
    return error


def calculate_error_rrmse(y_pred: np.ndarray, y_true: np.ndarray, do_print=False) -> float:
    """Calculating the Relative Root Mean Squared Error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    val0 = np.sum(np.abs(y_true - y_pred) ** 2) / y_pred.size
    val1 = np.sum(np.abs(y_pred) ** 2)
    error = np.sqrt(val0/val1)
    if do_print:
        print(f"Relative Root Mean squared error (RRMSE) = {error:.4f}")
    return error


def calculate_error_rmsre(y_pred: np.ndarray, y_true: np.ndarray, do_print=False) -> float:
    """Calculating the Root Mean Squared Relative ErrorArgs:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    val0 = np.sum((y_true - y_pred / y_pred) ** 2)
    error = np.sqrt(val0 / y_pred.size)
    if do_print:
        print(f"Root Mean Squared Relative Error (RMSRE) = {error:.4f}")
    return error


def compare_timestamps(true_labels: list, pred_labels: list, window=2) -> [float, float, list, list]:
    """ This function compares the timestamps of the predicted classes and the true classes and returns TP, FP, FN and
    new arrays which only contain the classes that have matched timestamps in both arrays. The function should be used
    before plotting a confusion matrix of the classes when working with actual data from the pipeline.
    Args:
        true_labels:    List with true labels
        pred_labels:    List with predicted labels
        window:         Window size for acceptance rate
    Returns:
        Metrics (F1 score, accuracy), new list
    """
    new_true = []
    new_pred = []
    FN = 0
    TP_same = 0
    TP_different = 0
    max_value = max(true_labels[0][-1], pred_labels[0][-1])
    for i in range(0, max_value):
        if i in true_labels[0]:
            found = False
            for j in range(i-int(window), i+int(window)+1):
                if j in pred_labels[0]:
                    pos_true = true_labels[0].index(i)
                    pos_pred = pred_labels[0].index(j)
                    new_true.append(true_labels[1][pos_true])
                    new_pred.append(pred_labels[1][pos_pred])
                    if true_labels[1][pos_true] == pred_labels[1][pos_pred]:
                        TP_same += 1
                    else:
                        TP_different += 1
                    found = True
            if not found:
                FN += 1
    FP = len(pred_labels[1])-len(new_pred)
    TP = TP_same+TP_different
    f1_score = 2 * TP / (2*TP+FP+FN)
    accuracy = TP / (TP+FP+FN)
    return f1_score, accuracy, new_true, new_pred


###################################### ELECTRICAL METRICS ######################################
def calculate_total_harmonics_distortion(freq: np.ndarray, spectral: np.ndarray,
                                         fsine: float, N_harmonics=4) -> [float, np.ndarray]:
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


def calculate_cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray, do_print=False) -> float:
    """Calculating the Total Harmonics Distortion (THD) of spectral input
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
        do_print:   Printing the value into terminal
    Returns:
        Float value with error
    """
    out = correlate(y_pred / np.linalg.norm(y_pred), y_true / np.linalg.norm(y_true),
                    'full', 'auto')
    cor_value = float(out[y_true.size])
    if do_print:
        print(f"\t Similarity coefficient = {100 * cor_value:.2f} %")
    return cor_value
