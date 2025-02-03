from os.path import abspath
import numpy as np

import denspp.offline.nsp.plot_nsp
from denspp.offline.pipeline.pipeline_cmds import PipelineCMD
from denspp.offline.pipeline.pipeline_signal import PipelineSignal
from denspp.offline.analog.amplifier.pre_amp import PreAmp, SettingsAMP
from denspp.offline.analog.adc import SettingsADC
from denspp.offline.analog.adc.adc_sar import SARADC as ADC0
from denspp.offline.digital.dsp import DSP, SettingsDSP
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA
from denspp.offline.digital.fex import FeatureExtraction, SettingsFeature
from denspp.offline.digital.cluster import Clustering, SettingsCluster
from denspp.offline.nsp.spike_analyse import calc_spiketicks


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
    # 20e3 für 40 samples per frame
    # 32e3 for 64 samples per frame

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
        mdict = {"fs_ana": self.signals.fs_ana,
                 "fs_adc": self.signals.fs_adc,
                 "fs_dig": self.signals.fs_dig,
                 "u_in": self.signals.u_in,
                 "x_spk": self.signals.x_spk,
                 "x_lfp": self.signals.x_lfp,
                 "frames_out": self.signals.frames_align[0],
                 "frames_pos": self.signals.frames_align[1],
                 "frames_id": self.signals.frames_align[2]}
        return mdict

    def do_plotting(self, data: PipelineSignal, channel: int) -> None:
        """Function to plot results"""
        import offline.pipeline.plot_pipeline as plt_neuro

        path2save = self.path2save
        # --- Spike Sorting output
        plt_neuro.plot_pipeline_afe(data, channel, path=path2save)
        plt_neuro.plot_transient_highlight_spikes(data, channel, path=path2save)
        plt_neuro.plot_transient_highlight_spikes(data, channel, path=path2save, time_cut=[10, 12])
        plt_neuro.plot_pipeline_frame_sorted(data, channel, path=path2save)
        plt_neuro.plot_pipeline_results(data, channel, path=path2save)

        # --- NSP block
        offline.nsp.plot_nsp.plot_nsp_ivt(data, channel, path=path2save)
        offline.nsp.plot_nsp.plot_firing_rate(data, channel, path=path2save)
        # plt_neuro.results_correlogram(data, channel, path=path2save)
        # plt_neuro.results_cluster_amplitude(data, channel, path=path2save)

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        u_inn = np.array(self.__preamp0.vcm)
        # ---- Analogue Front End Module ----
        self.signals.u_pre, _ = self.__preamp0.pre_amp_chopper(uinp, u_inn)
        self.signals.x_adc, _, self.signals.u_quant = self.__adc.adc_ideal(self.signals.u_pre)
        # ---- Digital Pre-processing ----
        self.signals.x_lfp = self.__dsp0.filter(self.signals.x_adc)
        self.signals.x_spk = self.__dsp1.filter(self.signals.x_adc)
        # ---- Spike detection incl. thresholding ----
        x_dly = self.__sda.time_delay(self.signals.x_spk)
        self.signals.x_sda = self.__sda.sda_smooth(self.__sda.sda_neo(self.signals.x_spk))
        self.signals.x_thr = self.__sda.thres_blackrock(self.signals.x_sda)
        (self.signals.frames_orig, self.signals.frames_align) = self.__sda.frame_generation(
            x_dly, self.signals.x_sda, self.signals.x_thr
        )
        # ---- Feature Extraction  ----
        self.signals.features = self.__fe.pca(self.signals.frames_align[0])
        # ---- Clustering | Classification ----
        self.signals.frames_align[2] = self.__cl.init(self.signals.features)
        self.signals.spike_ticks = calc_spiketicks(self.signals.frames_align)

        # --- Saving clustering model
        self.__cl.save_model_to_file('cluster_model', self.path2save)
