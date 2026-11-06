import torch
from torch import Tensor, eq, ne, sum
import torchmetrics.functional as FM


def calculate_number_true_predictions(pred: Tensor, true: Tensor) -> Tensor:
    """Calculate the total number of correctly predicted values.

    Args:
        pred: Tensor containing predicted labels from the model.
        true: Tensor containing ground truth labels.

    Returns:
        Tensor containing the count of matching predictions.
    """
    assert pred.shape == true.shape, "Dimension / shape mismatch"
    return sum(eq(pred, true))


def calculate_number_false_predictions(pred: Tensor, true: Tensor) -> Tensor:
    """Calculate the total number of incorrectly predicted values.

    Args:
        pred: Tensor containing predicted labels from the model.
        true: Tensor containing ground truth labels.

    Returns:
        Tensor containing the count of mismatched predictions.
    """
    assert pred.shape == true.shape, "Dimension / shape mismatch"
    return sum(ne(pred, true))


def _infer_num_classes(pred: Tensor, true: Tensor, num_classes: int | None) -> int:
    """Infer or validate the number of classes for multiclass metric calculation.

    Args:
        pred: Tensor containing predicted labels.
        true: Tensor containing ground truth labels.
        num_classes: Explicit class count, or None to infer from maximum label values.

    Returns:
        Integer number of classes, clamped to at least 2.
    """
    if num_classes is not None:
        return max(2, num_classes)
    if pred.numel() == 0:
        return 2
    return max(2, int(max(pred.max(), true.max()).item()) + 1)


def calculate_precision(true: Tensor, pred: Tensor, num_classes: int | None = None) -> Tensor:
    """Calculate the micro-averaged precision metric.

    Args:
        true: Tensor containing ground truth labels.
        pred: Tensor containing predicted labels from the model.
        num_classes: Optional number of classes for evaluation.

    Returns:
        Tensor scalar containing the precision score.
    """
    assert pred.shape == true.shape, "Dimension / shape mismatch"
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    n_cls = _infer_num_classes(pred, true, num_classes)
    return FM.precision(
        pred.long(), true.long(), task="multiclass", num_classes=n_cls, average="micro"
    )


def calculate_recall(pred: Tensor, true: Tensor, num_classes: int | None = None) -> Tensor:
    """Calculate the micro-averaged recall metric.

    Args:
        pred: Tensor containing predicted labels from the model.
        true: Tensor containing ground truth labels.
        num_classes: Optional number of classes for evaluation.

    Returns:
        Tensor scalar containing the recall score.
    """
    assert pred.shape == true.shape, "Dimension / shape mismatch"
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    n_cls = _infer_num_classes(pred, true, num_classes)
    return FM.recall(
        pred.long(), true.long(), task="multiclass", num_classes=n_cls, average="micro"
    )


def calculate_fbeta(
    pred: Tensor, true: Tensor, beta: float = 1.0, num_classes: int | None = None
) -> Tensor:
    """Calculate the micro-averaged F-beta metric.

    Args:
        pred: Tensor containing predicted labels from the model.
        true: Tensor containing ground truth labels.
        beta: Weighting factor between precision and recall (default: 1.0 for F1-score).
        num_classes: Optional number of classes for evaluation.

    Returns:
        Tensor scalar containing the F-beta score.
    """
    assert pred.shape == true.shape, "Dimension / shape mismatch"
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    n_cls = _infer_num_classes(pred, true, num_classes)
    return FM.fbeta_score(
        pred.long(),
        true.long(),
        beta=float(beta),
        task="multiclass",
        num_classes=n_cls,
        average="micro",
    )