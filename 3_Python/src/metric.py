import numpy as np

class Metric():
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

        # Metrics
        self.sse = None
        self.dr = None           # ???
        self.ca = None           # compression accuracy
        self.cr = None           # compression ratio

    def calculate_snr(self, yin: np.ndarray, ymean: np.ndarray):
        A = np.sum(np.square(yin))
        B = np.sum(np.square(ymean - yin))
        outdB = 10 * np.log10(A/B)
        return outdB

    def metric_fec(self) -> None:
        pass

    def metric_sda(self, Xsoll: np.ndarray) -> None:
        # TODO: Metrik prüfen
        print("... Calculation of metrics with labeled informations")
        TP = 0  # number of true positive
        TN = 0  # number of true negative
        FP = 0  # number of false positive
        FN = 0  # number of false negative
        tol = 2 * self.frame_length

        for idxX in self.x_pos:
            for idxY in Xsoll:
                dX = idxY - idxX
                # --- Decision tree
                if np.abs(dX) < tol:
                    TP += 1
                    break
                elif dX > 2 * tol:
                    FP += 1
                    break

        FN = Xsoll.size - TP - FP
        TN = np.floor(self.x_pos.size) - TP

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

        print("... Detected activities", self.x_pos.size, "of", Xsoll.size)
        print("... Metrics (SDA):", np.round(FPR, 3), "(FPR)", np.round(FNR, 3), "(FNR)", np.round(TPR, 3), "(TPR)",
              np.round(TNR, 3), "(TNR)")
        print("... Metrics (SDA):", np.round(PPV, 3), "(PPV)", np.round(NPV, 3), "(NPV)")
        print("... Accuracy (SDA):", np.round(Accuracy, 3))

        # Ausgabe
        self.ca = Accuracy
        self.dr = self.x_adc.size / self.frame_length
