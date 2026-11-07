import numpy as np


def calculate_error_mbe(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with mean bias error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sum(y_pred - y_true) / y_pred.size)
    else:
        return y_pred - y_true


def calculate_error_mae(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with mean absolute error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sum(np.abs(y_pred - y_true)) / y_pred.size)
    else:
        return float(np.abs(y_pred - y_true))


def calculate_error_mse(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with mean squared error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sum((y_pred - y_true) ** 2) / y_pred.size)
    else:
        return float(y_pred - y_true) ** 2


def calculate_error_mpe(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with mean percentage error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sum((y_true - y_pred) / y_true) / y_true.size)
    else:
        return float((y_true - y_pred) / y_true)


def calculate_error_mape(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with mean absolute percentage error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    if isinstance(y_true, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sum(np.abs(y_true - y_pred) / np.abs(y_true)) / y_true.size)
    else:
        return float(abs(y_true - y_pred) / abs(y_true))


def calculate_error_rae(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with relative absolute error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    mse = calculate_error_mse(y_pred, y_true)
    if isinstance(y_pred, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(np.sqrt(mse) / calculate_error_mae(np.zeros(shape=y_pred.shape), y_true))
    else:
        return float((mse**0.5) / calculate_error_mae(0.0, y_pred))


def calculate_error_rse(y_pred: float | np.ndarray, y_true: float | np.ndarray) -> float:
    """Calculating the distance-based metric with relative squared error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    mse = calculate_error_mse(y_pred, y_true)
    y_true_mean = np.mean(y_true)
    if isinstance(y_pred, np.ndarray):
        assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
        return float(mse / np.sum((y_pred - y_true_mean) ** 2) / y_pred.size)
    else:
        return float(mse / (y_pred - y_true_mean) ** 2)


def calculate_error_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculating the Root Mean Squared Error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
    return float(np.sqrt(np.sum(np.abs(y_pred - y_true) ** 2) / y_pred.size))


def calculate_error_rrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculating the Relative Root Mean Squared Error
    Args:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
    val0 = np.sum(np.abs(y_true - y_pred) ** 2) / y_pred.size
    val1 = np.sum(np.abs(y_pred) ** 2)
    return float(np.sqrt(val0 / val1))


def calculate_error_rmsre(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculating the Root Mean Squared Relative ErrorArgs:
        y_pred:     Numpy array or float value from prediction
        y_true:     Numpy array or float value from true label
    Returns:
        Float value with error
    """
    assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
    val0 = np.sum((y_true - y_pred / y_pred) ** 2)
    return float(np.sqrt(val0 / y_pred.size))
