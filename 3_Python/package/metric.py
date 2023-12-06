import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def calculate_snr(yin: np.ndarray, ymean: np.ndarray):
    """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform"""
    a0 = (np.max(ymean) - np.min(ymean)) ** 2
    b0 = np.sum((yin - ymean) ** 2)
    return 10 * np.log10(a0 / b0)


def calculate_prd(yin: np.ndarray, ymean: np.ndarray):
    """Calculating the root-mean-square difference in percentage (PRD)"""
    a0 = np.sum(np.square(yin - ymean))
    b0 = np.sum(np.squre(yin))
    return np.sqrt(a0 / b0) * 100

# This function compares the timestamps of the predicted classes and the true classes and returns TP, FP, FN and
# new arrays which only contain the classes that have matched timestamps in both arrays. The function should be used
# before plotting a confusion matrix of the classes when working with actual data from the pipeline.


def compare_timestamps(true_labels: list, pred_labels: list, window=2):
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
    f1_score = 2*(TP)/(2*(TP)+FP+FN)
    accuracy = TP/(TP+FP+FN)
    return TP, FP, FN, f1_score, accuracy, new_true, new_pred



class Metric:
    """Class for Determining the Metrics"""
    def __init__(self):
        self.cm = 0.0
        # Metrics
        self.sse = 0.0
        self.dr = 0.0           # ???
        self.ca = 0.0           # compression accuracy
        self.cr = 0.0           # compression ratio

    def __preprocess_sda0(self, x_ist: np.ndarray, x_soll: np.ndarray, tol: int):
        TP = 0  # number of true positive
        TN = 0  # number of true negative
        FP = 0  # number of false positive
        FN = 0  # number of false negative

        for idxX in x_ist:
            for idxY in x_soll:
                dX = idxY - idxX
                # --- Decision tree
                if np.abs(dX) < tol:
                    TP += 1
                    break
                elif dX > 2 * tol:
                    FP += 1
                    break

        FN = x_soll.size - TP - FP
        TN = np.floor(x_ist.size) - TP

        return TP, TN, FP, FN

    def __preprocess_sda1(self, x_ist: np.ndarray, x_soll: np.ndarray) -> [float, float, float, float]:
        # process the sda staff

        # Get the confusion matrix
        (TN, TP, FN, FP) = confusion_matrix()

        return TP, TN, FP, FN

    def __calculate_param(self, TP: float, TN: float, FP: float, FN: float) -> [float, float, float, float, float, float, float]:
        # False Positive rate - Probability of false alarm
        FPR = float(FP / (FP + TN))
        # False Negative rate - Miss rate
        FNR = float(FN / (FN + TP))
        # True Positive rate - Sensitivity
        TPR = float(TP / (TP + FN))
        # True Negative rate - Specificity
        TNR = float(TN / (TN + FP))
        # Positive predictive value
        PPV = float(TP / (TP + FP))
        # Negative predictive value
        NPV = float(TN / (TN + FN))
        # Accuracy
        acc = float((TP + TN) / (TP + TN + FN + FP))
        precision = float(TP/(TP + FP))

        return FPR, FNR, TPR, TNR, PPV, NPV, acc

    def metric_sda(self, xpred: np.ndarray, xtrue: np.ndarray) -> None:
        (TP, TN, FP, FN) = self.__preprocess_sda0(xpred, xtrue)
        (FPR, FNR, TPR, TNR, PPV, NPV, Acc) = self.__calculate_param(TP, TN, FP, FN)
        self.ca = Acc

        # Output of meta information
        print(f"... Detected activities {xpred.size} of {xtrue.size}")
        print(f"... Metrics (SDA): {FPR:.3f} (FPR) {FNR:.3f} (FNR) {TPR:.3f} (TPR) {TNR:.3f} (TNR)")
        print(f"... Metrics (SDA): {PPV:.3f} (PPV) {NPV:.3f} (NPV)")
        print(f"... Accuracy (SDA): {100 * Acc:.3f} %")

    def metric_fec(self, xpred: np.ndarray, ypred: np.ndarray, xtrue: np.ndarray, ytrue: np.ndarray) -> None:
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        self.cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot()
        plt.show()
