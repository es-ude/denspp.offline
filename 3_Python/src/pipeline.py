import numpy as np

from settings import Settings
from src.afe import AFE
from src.fec import FEC

import tensorflow.keras as nntf

#from src_ai.nn_pytorch import NeuralNetwork
from src_ai.nn_tensorflow import NeuralNetwork

class PipelineSpike (AFE, FEC, NeuralNetwork, nntf.Model):
    def __init__(self, settings: Settings):
        AFE.__init__(self, settings)
        FEC.__init__(self, settings)
        NeuralNetwork.__init__(self, 40)
        nntf.Model.__init__(self)

        # Settings for AI
        self.denoising_name = "dnn_dae_v2_TEST"
        self.denoising_model = nntf.models.Sequential()

        # Settings
        self.version = settings.version
        self.__mode_thres = settings.mode_thres
        self.__mode_frame = settings.mode_frame

        # Signals for Input and Output
        self.u_in = None
        self.u_lfp = None
        self.u_spk = None
        self.x_adc = None
        self.x_sda = None
        self.x_thr = None
        self.x_pos = None
        self.frames_orig = None
        self.frames_align = None
        self.features = None
        self.cluster_id = None
        self.cluster_no = None
        self.spike_ticks = None

        # Additional signals from AI pre-processing
        self.frames_denoised = None

        # Metrics
        self.sse = None
        self.dr = None           # ???
        self.ca = None           # compression accuracy
        self.cr = None           # compression ratio

    def initMod(self, typeRun) -> None:
        self.__version = typeRun
        if self.__version == 1:
            self.denoising_model = self.initPredict(self.denoising_name)

    def runPipeline(self, doCalc) -> None:
        if self.version == 0:
            self.__spikesorting_pipeline_v0(doCalc)
        elif self.version == 1:
            self.__spikesorting_pipeline_v1(doCalc)
        else:
            ValueError("System error: Pipeline version is not available!")

    def metric_afe(self, Xsoll: np.ndarray) -> None:
        # TODO: Metrik prüfen
        print("... Calculation of metrics with labeled informations")
        TP = 0  # number of true positive
        TN = 0  # number of true negative
        FP = 0  # number of false positive
        FN = 0  # number of false negative
        tol = 2*self.frame_length

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


    def __spikesorting_pipeline_v0(self, doCalc) -> None:
        # ----- Analogue Front End Module  -----
        (self.u_spk, self.u_lfp) = self.pre_amp(self.u_in)
        self.x_adc = self.adc_nyquist(self.u_spk, doCalc[0])

        # --- Digital Pre-processing ----
        self.x_filt = self.dig_filt(self.x_adc)

        # --- Spike detection incl. thresholding
        (self.x_sda, self.x_thr) = self.spike_detection(self.x_filt, self.__mode_thres, doCalc[1])
        self.x_dly = self.time_delay_dig(self.x_filt)
        (self.frames_orig, self.x_pos) = self.frame_generation(self.x_dly, self.x_sda, self.x_thr)
        self.frames_align = self.frame_aligning(self.frames_orig, self.__mode_frame, doCalc[2])

        # ----- Feature Extraction and Classification Module -----
        self.features = self.fe_pca(self.frames_align)
        (self.cluster_id, self.cluster_no, self.sse) = self.cluster_kmeans(self.features)
        self.spike_ticks = self.calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id, self.cluster_no)

    def __spikesorting_pipeline_v1(self, doCalc) -> None:
        # ----- Analogue Front End Module  -----
        (self.u_spk, self.u_lfp) = self.pre_amp(self.u_in)
        self.x_adc = self.adc_nyquist(self.u_spk, doCalc[0])

        # --- Digital Pre-processing ----
        self.x_filt = self.dig_filt(self.x_adc)

        # --- Spike detection incl. thresholding
        (self.x_sda, self.x_thr) = self.spike_detection(self.x_filt, self.__mode_thres, doCalc[1])
        self.x_dly = self.time_delay_dig(self.x_filt)
        (self.frames_orig, self.x_pos) = self.frame_generation(self.x_dly, self.x_sda, self.x_thr)
        self.frames_align = self.frame_aligning(self.frames_orig, self.__mode_frame, doCalc[2])

        # --- Adding denoising
        val_max = 48
        self.frames_denoised = self.denoising_model.predict(self.frames_align/val_max)
        self.frames_denoised = self.frames_denoised * val_max

        # ----- Feature Extraction and Classification Module -----
        self.features = self.fe_pca(self.frames_denoised)
        (self.cluster_id, self.cluster_no, self.sse) = self.cluster_kmeans(self.features)
        self.spike_ticks = self.calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id, self.cluster_no)
