import numpy as np


def calculate_snr(yin: np.ndarray, ymean: np.ndarray) -> np.ndarray:
    """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform"""
    a0 = (np.max(ymean) - np.min(ymean)) ** 2
    b0 = np.sum((yin - ymean) ** 2)
    return 10 * np.log10(a0 / b0)


def calculate_prd(yin: np.ndarray, ymean: np.ndarray):
    """Calculating the root-mean-square difference in percentage (PRD)"""
    a0 = np.sum(np.square(yin - ymean))
    b0 = np.sum(np.square(yin))
    return np.sqrt(a0 / b0) * 100


def _error_mbe(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean bias error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(y_pred - y_true) / y_pred.size)
    else:
        error = y_pred - y_true
    return error


def _error_mae(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean absolute error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(np.abs(y_pred - y_true)) / y_pred.size)
    else:
        error = float(np.abs(y_pred - y_true))
    return error


def _error_mse(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean squared error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum((y_pred - y_true) ** 2) / y_pred.size)
    else:
        error = float(y_pred - y_true) ** 2
    return error


def _error_rae(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with relative absolute error"""
    mse = _error_mse(y_pred, y_true)
    if isinstance(mse, np.ndarray):
        error = float(np.sqrt(mse) / _error_mae(np.zeros(shape=y_pred.shape), y_true))
    else:
        error = float((mse ** 0.5) / _error_mae(0.0, y_pred))
    return error


def _error_rse(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with relative squared error"""
    mse = _error_mse(y_pred, y_true)
    y_true_mean = np.mean(y_true)
    if isinstance(mse, np.ndarray):
        error = float(mse / np.sum((y_pred - y_true_mean) ** 2) / y_pred.size)
    else:
        error = float(mse / (y_pred - y_true_mean) ** 2)
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
            for j in range(i-window, i+window+1):
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
