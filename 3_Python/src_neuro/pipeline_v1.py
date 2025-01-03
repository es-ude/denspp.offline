from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD
from package.pipeline_signal import PipelineSignal
from package.nsp.spike_analyse import calc_spiketicks
from package.analog.amplifier.pre_amp import PreAmp, SettingsAMP
from package.analog.adc import SettingsADC
from package.analog.adc.adc_sar import SARADC as ADC0
from package.digital.dsp import DSP, SettingsDSP
from package.digital.sda import SpikeDetection, SettingsSDA
from package.digital.fex import FeatureExtraction, SettingsFeature
from package.digital.cluster import Clustering, SettingsCluster


class _SettingsPipe:
    """Settings class for setting-up the pipeline"""
    def __init__(self, fs: float):
        self.SettingsAMP.fs_ana = fs
        self.SettingsADC.fs_ana = fs

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
        type_out="signed",
        dvref=0.1,
        fs_ana=0.0,
        fs_dig=20e3, osr=1, Nadc=12
    )

    # --- Digital filtering for ADC output and CIC
    SettingsDSP_LFP = SettingsDSP(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=2, f_filt=[0.1, 100],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
    )
    SettingsDSP_SPK = SettingsDSP(
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

        settings = _SettingsPipe(fs_ana)
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
        import package.plot.plot_pipeline as plt_neuro

        path2save = self.path2save
        # --- Spike Sorting output
        plt_neuro.results_afe1(data, channel, path=path2save)
        plt_neuro.results_afe_sorted(data, channel, path=path2save)
        plt_neuro.results_afe_sorted(data, channel, path=path2save, time_cut=[10, 12])
        plt_neuro.results_fec(data, channel, path=path2save)
        plt_neuro.results_paper(data, channel, path=path2save)

        # --- NSP block
        plt_neuro.results_ivt(data, channel, path=path2save)
        #plt_neuro.results_firing_rate(data, channel, path=path2save)
        # plt_neuro.results_correlogram(data, channel, path=path2save)
        # plt_neuro.results_cluster_amplitude(data, channel, path=path2save)

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        # ---- Analogue Front End Module ----
        self.signals.u_pre, _ = self.__preamp0.pre_amp_chopper(uinp, np.array(self.__preamp0.vcm))
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
            self.signals.frames_align[2] = self.__cl.init(self.signals.features)
            self.signals.spike_ticks = calc_spiketicks(self.signals.frames_align)

            # --- Saving clustering model
            self.__cl.save_model_to_file('cluster_model', self.path2save)
