from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, sum, eq


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


def calculate_fbeta(pred: Tensor, true: Tensor, beta: float=1.0) -> Tensor:
    """Function for determining the precision metric
    Args:
        pred:   Tensor with predicted values from model
        true:   Tensor with true labels from dataset
        beta:   Beta value for getting Fbeta metric
    Return
        Tensor with metrics [precision]
    """
    return precision_recall_fscore_support(true, pred, beta=beta, average="micro", warn_for=tuple())[2]
