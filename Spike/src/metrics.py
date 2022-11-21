import numpy as np

class QualityParam:
    dr = None
    ca = None
    cr = None

def metric_afe(labeling, tol):
    TP = 0  # number of true positive
    TN = 0  # number of true negative
    FP = 0  # number of false positive
    FN = 0  # number of false negative

    XposIn = find(single(Xin) == 1)
    for idxX in range(0, XposIn.size):
        for idxY in range(0, Xchk.size):
            dX = XposIn(idxX) - Xchk(idxY)
            # --- Decision tree
            if (abs(dX) <= tol):
                TP = TP + 1
                break
            elif idxY == Xchk.size:
                FP = FP + 1
                break

    FN = Xchk.size - TP - FP
    TN = floor(length(Xin) / SpikeAFE.XWindowLength) - TP

    # --- Output parameters
    # False Positive rate - Probability of false alarm
    FPR = FP / (FP + TN)
    # False Negative rate - Miss rate
    FNR = FN / (FN + TP)
    # True Positive rate - Sensitivity
    TPR = TP / (TP + FN)
    # True Negative rate - Specificity
    TNR = TN / (TN + FP)
    # Positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)

    Accuracy = (TP + TN) / (TP + TN + FN + FP)

    return (FPR, FNR, TPR, TNR, PPV, NPV, Accuracy)