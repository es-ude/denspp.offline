from dataclasses import dataclass

@dataclass
class MetricTimestamps:
    f1_score: float
    TP: int
    FP: int
    FN: int


def compare_timestamps(
        true_labels: list, pred_labels: list, window: int=2
) -> MetricTimestamps:
    """ This function compares the timestamps of the predicted classes and the true classes and returns TP, FP, FN and
    new arrays which only contain the classes that have matched timestamps in both arrays. The function should be used
    before plotting a confusion matrix of the classes when working with actual data from the pipeline.
    Args:
        true_labels:    List with true labels
        pred_labels:    List with predicted labels
        window:         Window size for acceptance rate
    Returns:
        Class MetricTimeStamps with metrics
    """
    new_pred = []
    false_negative = 0
    true_positive_same = 0
    true_positive_diff = 0

    for i in range(0, max(true_labels[-1], pred_labels[-1])+1):
        if i in true_labels:
            found = False
            for j in range(i-int(window), i+int(window)+1):
                if j in pred_labels:
                    pos_true = true_labels.index(i)
                    pos_pred = pred_labels.index(j)
                    new_pred.append(pred_labels[pos_pred])
                    if true_labels[pos_true] == pred_labels[pos_pred]:
                        true_positive_same += 1
                    else:
                        true_positive_diff += 1
                    found = True
            if not found:
                false_negative += 1

    if len(pred_labels) - len(true_labels) > 0:
        false_positive = len(pred_labels) - len(true_labels)
    else:
        false_positive = 0
    true_positive = true_positive_same + true_positive_diff

    f1_score = true_positive / (true_positive + false_positive + false_negative)
    return MetricTimestamps(
        f1_score=f1_score,
        FN=false_negative,
        FP=false_positive,
        TP=true_positive
    )
