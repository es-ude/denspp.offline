import os, shutil

from src.amp.preamp import PreAmp
from src.adc.adc import ADC
from src.dsp.dsp import DSP
from src.sda.sda import SDA
from src.feature.fec import FEC
from src.metric import Metric
from settings import Settings
from src.data_call import DataHandler

class PipelineSpike (Metric):
    def __init__(self, settings: Settings):
        self.preamp0 = PreAmp(settings, settings.f_filt_ana)
        self.preamp1 = PreAmp(settings, settings.f_filt_spk)
        self.preamp2 = PreAmp(settings, settings.f_filt_lfp)
        self.adc = ADC(settings)
        self.dsp0 = DSP(settings, settings.f_filt_spk)
        self.dsp1 = DSP(settings, settings.f_filt_lfp)
        self.sda = SDA(settings)
        self.fec = FEC(settings)
        Metric.__init__(self, settings.x_window_length)

        self.fs_ana = settings.fs_ana
        self.fs_adc = settings.fs_adc
        self.__scaling_metric = self.fs_adc / self.fs_ana

        # Settings for saving figures, logs and results
        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None

        # Settings for AI
        self.path2model = "models"
        self.denoising_model = None

        # Settings
        self.version = settings.version
        self.__mode_thres = settings.mode_thres
        self.__mode_frame = settings.mode_frame

        # Signals for Input and Output
        self.u_in = None
        self.u_pre = None
        self.u_spk = None
        self.u_lfp = None
        self.x_adc = None
        self.x_spk = None
        self.x_lfp = None
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

            # self.metric_fec(path2save, xposIst, SpikeSorting.cluster_id, xposSoll, dataIn.cluster_id)

    def saving_results(self, name: str) -> str:
        if not os.path.exists(self.path2runs):
            os.mkdir(self.path2runs)

        path2figure = os.path.join(self.path2runs, name)
        if not os.path.exists(path2figure):
            os.mkdir(path2figure)

        # --- Copy settings into this folder
        shutil.copy(src="settings.py", dst=path2figure)
        self.path2figure = path2figure

        return path2figure

    def initMod(self, typeRun) -> None:
        self.__version = typeRun
        if self.__version == 1:
            self.denoising_model = self.initPredict(self.denoising_name)

    def runPipeline(self) -> None:
        if self.version == 0:
            self.__spikesorting_pipeline_v0()
        elif self.version == 1:
            self.__spikesorting_pipeline_v1()
        else:
            ValueError("System error: Pipeline version is not available!")

    def __spikesorting_pipeline_v0(self) -> None:
        # ----- Analogue Front End Module  -----
        self.u_pre = self.preamp0.pre_amp(self.u_in)
        self.x_adc = self.adc.adc_nyquist(self.u_pre, True)

        # --- Digital Pre-processing ----
        # self.x_lfp = self.dsp1.dig_filt_iir(self.x_adc)
        self.x_spk = self.dsp0.dig_filt_iir(self.x_adc)
        self.x_dly = self.dsp0.time_delay(self.x_spk, Settings.x_offset[0])

        # --- Spike detection incl. thresholding
        (self.x_sda, self.x_thr) = self.sda.sda(self.x_spk, self.__mode_thres)
        # self.x_sda = self.dsp0.dig_filt_iir(self.x_sda)
        (self.frames_orig, self.x_pos) = self.sda.frame_generation(self.x_dly, self.x_sda, self.x_thr)
        self.frames_align = self.sda.frame_aligning(self.frames_orig, self.__mode_frame)

        # ----- Feature Extraction and Classification Module -----
        self.features = self.fec.fe_pca(self.frames_align)
        (self.cluster_id, self.cluster_no, self.sse) = self.fec.cluster_kmeans(self.features)
        self.spike_ticks = self.fec.calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id)

    def __spikesorting_pipeline_v1(self) -> None:
        # ----- Analogue Front End Module  -----
        (self.u_pre) = self.preamp0.pre_amp(self.u_in)
        self.x_adc = self.adc.adc_nyquist(self.u_pre, True)

        # --- Digital Pre-processing ----
        # self.x_lfp = self.dsp1.dig_filt_iir(self.x_adc)
        self.x_spk = self.dsp0.dig_filt_iir(self.x_adc)
        self.x_dly = self.dsp0.time_delay(self.x_spk, Settings.x_offset[0])

        # --- Spike detection incl. thresholding
        (self.x_sda, self.x_thr) = self.sda.sda(self.x_spk, self.__mode_thres)
        (self.frames_orig, self.x_pos) = self.sda.frame_generation(self.x_spk, self.x_sda, self.x_thr)
        self.frames_align = self.sda.frame_aligning(self.frames_orig, self.__mode_frame)

        # --- Adding denoising
        val_max = 48
        self.frames_denoised = self.denoising_model.predict(self.frames_align/val_max)
        self.frames_denoised = self.frames_denoised * val_max

        # ----- Feature Extraction and Classification Module -----
        self.features = self.fec.fe_pca(self.frames_denoised)
        (self.cluster_id, self.cluster_no, self.sse) = self.fec.cluster_kmeans(self.features)
        self.spike_ticks = self.fec.calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id)
