import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.data_call import DataHandler

class Metric():
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

        self.cm = None
        # Metrics
        self.sse = None
        self.dr = None           # ???
        self.ca = None           # compression accuracy
        self.cr = None           # compression ratio

    def check_label(self, dataIn: DataHandler) -> None:
        if not dataIn.label_exist:
            pass
        else:
            print("... Calculation of metrics with labeled informations")
            path2save = self.path2figure

            x_ist = self.x_pos
            x_soll = dataIn.spike_xpos * self.__scaling_metric
            self.dr = self.x_adc.size / self.frame_length
            self.metric_sda(path2save, x_ist, x_soll)
    def calculate_snr(self, yin: np.ndarray, ymean: np.ndarray):
        A = np.sum(np.square(yin))
        B = np.sum(np.square(ymean - yin))
        outdB = 10 * np.log10(A/B)
        return outdB

    # TODO: Metrik prüfen
    def __preprocess_sda0(self, x_ist: np.ndarray, x_soll: np.ndarray):
        TP = 0  # number of true positive
        TN = 0  # number of true negative
        FP = 0  # number of false positive
        FN = 0  # number of false negative
        tol = 2 * self.frame_length

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

    def metric_sda(self, path: str, xpred: np.ndarray, xtrue: np.ndarray) -> None:
        (TP, TN, FP, FN) = self.__preprocess_sda0(xpred, xtrue)
        (FPR, FNR, TPR, TNR, PPV, NPV, Acc) = self.__calculate_param(TP, TN, FP, FN)
        self.ca = Acc

        # Output of meta information
        print(f"... Detected activities {xpred.size} of {xtrue.size}")
        print(f"... Metrics (SDA): {FPR:.3f} (FPR) {FNR:.3f} (FNR) {TPR:.3f} (TPR) {TNR:.3f} (TNR)")
        print(f"... Metrics (SDA): {PPV:.3f} (PPV) {NPV:.3f} (NPV)")
        print(f"... Accuracy (SDA): {100 * Acc:.3f} %")

        # Saving plots
        do_plot = False
        if path and do_plot:
            format = ['eps', 'svg']
            name = "pipeline_sda"
            path2fig = os.path.join(path, name)

            for idx, form in enumerate(format):
                file_name = path2fig + '.' + form
                plt.savefig(file_name, format=form)

    def metric_fec(self, xpred: np.ndarray, ypred: np.ndarray, xtrue: np.ndarray, ytrue: np.ndarray, path: str) -> None:
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        self.cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot()
        plt.show()

        if path:
            format = ['eps', 'svg']
            name = "pipeline_cm"
            path2fig = os.path.join(path, name)

            for idx, form in enumerate(format):
                file_name = path2fig + '.' + form
                plt.savefig(file_name, format=form)
