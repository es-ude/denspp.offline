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
