from os.path import abspath
import numpy as np

from denspp.offline.pipeline.pipeline_cmds import PipelineCMD
from denspp.offline.pipeline.pipeline_signal import PipelineSignal
from denspp.offline.nsp.spike_analyse import calc_spiketicks
from denspp.offline.analog.amplifier.pre_amp import PreAmp, SettingsAMP
from denspp.offline.analog.adc import SettingsADC
from denspp.offline.analog.adc.adc_sar import SuccessiveApproximation as ADC0
from denspp.offline.digital.dsp import DSP, SettingsFilter
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA
from denspp.offline.digital.fex import FeatureExtraction, SettingsFeature
from denspp.offline.digital.cluster import Clustering, SettingsCluster


class SettingsPipe:
    def __init__(self, fs: float, vss: float=-0.6, vdd: float=0.6) -> None:
        """Settings class for setting-up the pipeline
        :param fs:      Sampling frequency [Hz]
        :param vss:     Negative Supply Voltage [V]
        :param vdd:     Positive Supply Voltage [V]
        """
        self.SettingsAMP.fs_ana = fs
        self.SettingsADC.fs_ana = fs
        self.__overwrite_power_supply(vss, vdd)

    def __overwrite_power_supply(self, vss: float, vdd: float) -> None:
        a = [method for method in dir(self) if 'Settings' in method and '__' not in method]
        for setting in a:
            set0 = getattr(self, setting)
            set0.vss = vss
            set0.vdd = vdd

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=0.0,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6, noise_en=True,
        f_chop=10e3,
        noise_edev=100e-9
    )

    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        is_signed=True,
        dvref=0.1,
        fs_ana=0.0,
        fs_dig=20e3, osr=1, Nadc=12
    )

    # --- Digital filtering for ADC output and CIC
    SettingsDSP_LFP = SettingsFilter(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=2, f_filt=[0.1, 100],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
    )
    SettingsDSP_SPK = SettingsFilter(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=2, f_filt=[200, 8e3],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
    )

    # --- Options for Spike Detection and Frame Aligning
    SettingsSDA = SettingsSDA(
        fs=SettingsADC.fs_adc, dx_sda=[1],
        mode_align=1,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.1e-3, 0.1e-3],
        t_dly=0.4e-3,
        window_size=7,
        thr_gain=1.0,
        thr_min_value=100.0
    )

    # --- Options for MachineLearning Part
    SettingsFE = SettingsFeature(
        no_features=3
    )
    SettingsCL = SettingsCluster(
        type="kMeans",
        no_cluster=3
    )


class Pipeline(PipelineCMD):
    """Processing Pipeline for analysing invasive neural activities"""
    def __init__(self, fs_ana: float):
        super().__init__()
        self._path2pipe = abspath(__file__)
        self.generate_folder('runs', '_neuro')

        settings = SettingsPipe(fs_ana)
        self.signals = PipelineSignal()
        self.signals.fs_ana = settings.SettingsADC.fs_ana
        self.signals.fs_adc = settings.SettingsADC.fs_adc
        self.signals.fs_dig = settings.SettingsADC.fs_dig

        self.__preamp0 = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__dsp0 = DSP(settings.SettingsDSP_LFP)
        self.__dsp1 = DSP(settings.SettingsDSP_SPK)
        self.__sda = SpikeDetection(settings.SettingsSDA)
        self.__fe = FeatureExtraction(settings.SettingsFE)
        self.__cl = Clustering(settings.SettingsCL)

    def prepare_saving(self) -> dict:
        """Getting processing data of selected signals"""
        mdict = {"fs_adc": self.signals.fs_adc,
                 "fs_dig": self.signals.fs_dig,
                 "x_adc": np.array(self.signals.x_adc, dtype=np.int16),
                 "x_spk": np.array(self.signals.x_spk, dtype=np.int16),
                 "frames_out": self.signals.frames_align[0],
                 "frames_pos": self.signals.frames_align[1],
                 "frames_id": self.signals.frames_align[2]}
        return mdict

    def do_plotting(self, data: PipelineSignal, channel: int) -> None:
        """Function to plot results"""
        import denspp.offline.pipeline.plot_pipeline as plt_neuro

        plt_neuro.plot_pipeline_frame_sorted(data, channel, path=self.path2save)
        plt_neuro.plot_pipeline_results(data, channel, path=self.path2save)

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        # ---- Analogue Front End Module ----
        self.signals.u_pre = self.__preamp0.pre_amp_chopper(uinp, np.array(self.__preamp0.vcm))['out']
        self.signals.x_adc, _, self.signals.u_quant = self.__adc.adc_ideal(self.signals.u_pre)
        # ---- Digital Pre-processing ----
        self.signals.x_lfp = self.__dsp0.filter(self.signals.x_adc)
        self.signals.x_spk = self.__dsp1.filter(self.signals.x_adc)
        # ---- Spike detection incl. thresholding ----
        x_dly = self.__sda.time_delay(self.signals.x_spk)
        self.signals.x_sda = self.__sda.sda_spb(self.signals.x_spk, [200, 2e3])
        self.signals.x_thr = self.__sda.thres_blackrock(self.signals.x_sda)
        self.signals.frames_orig, self.signals.frames_align = self.__sda.frame_generation(
            x_dly, self.signals.x_sda, self.signals.x_thr
        )

        # ---- Feature Extraction  ----
        if self.signals.frames_align[1].size == 0:
            print("No frames available!")
        else:
            self.signals.features = self.__fe.pca(self.signals.frames_align[0])
            # ---- Clustering | Classification ----
            (self.signals.frames_align[2]) = self.__cl.cluster_kmeans(self.signals.features)
            self.signals.spike_ticks = calc_spiketicks(self.signals.frames_align)
