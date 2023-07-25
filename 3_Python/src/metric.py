import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.data_call import DataHandler

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

# TODO: Metrik-Klasse ausbauen
class Metric():
    def __init__(self, path2save: str):
        self.path2figure = path2save

        self.cm = None
        # Metrics
        self.sse = None
        self.dr = None           # ???
        self.ca = None           # compression accuracy
        self.cr = None           # compression ratio

    def __save_figure(self, name: str) -> None:
        """Saving the selected figure/plot"""
        format = ['eps', 'svg']
        path2fig = os.path.join(self.path2figure, name)

        for idx, form in enumerate(format):
            file_name = path2fig + '.' + form
            plt.savefig(file_name, format=form)

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

        self.__save_figure("metric_sda")
