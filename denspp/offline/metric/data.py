import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, sum, eq


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


def calculate_number_true_predictions(pred: Tensor, true: Tensor) -> Tensor:
    """Function for determining the true predicted values
    Args:
        pred:   Tensor with predicted values from model
        true:   Tensor with true labels from dataset
    Return
        Tensor with metric
    """
    return sum(eq(pred, true))


def calculate_precision(pred: Tensor, true: Tensor) -> Tensor:
    """Function for determining the precision metric
    Args:
        pred:   Tensor with predicted values from model
        true:   Tensor with true labels from dataset
    Return
        Tensor with metrics [precision]
    """
    return precision_recall_fscore_support(true, pred, average="micro", warn_for=tuple())[0]


def calculate_recall(pred: Tensor, true: Tensor) -> Tensor:
    """Function for determining the precision metric
    Args:
        pred:   Tensor with predicted values from model
        true:   Tensor with true labels from dataset
    Return
        Tensor with metrics [precision]
    """
    return precision_recall_fscore_support(true, pred, average="micro", warn_for=tuple())[1]


def calculate_fbeta(pred: Tensor, true: Tensor, beta=1.0) -> Tensor:
    """Function for determining the precision metric
    Args:
        pred:   Tensor with predicted values from model
        true:   Tensor with true labels from dataset
        beta:   Beta value for getting Fbeta metric
    Return
        Tensor with metrics [precision]
    """
    return precision_recall_fscore_support(true, pred, beta=beta, average="micro", warn_for=tuple())[2]
