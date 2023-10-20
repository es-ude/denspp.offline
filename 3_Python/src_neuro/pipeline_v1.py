import os
import shutil
import numpy as np

from package.pipeline_signals import PipelineSignal
from package.data_call import SettingsDATA
from package.pre_amp.preamp import PreAmp, SettingsAMP
from package.adc.adc_basic import SettingsADC
from package.adc.adc_sar import ADC_SAR as ADC0
from package.dsp.dsp import DSP, SettingsDSP
from package.dsp.sda import SpikeDetection, SettingsSDA
from package.dsp.fex import FeatureExtraction, SettingsFeature
from package.dsp.cluster import Clustering, SettingsCluster
from package.nsp import calc_spiketicks


# --- Configuring the src_neuro
class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        # path='C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten',
        path='C:\GitHub\spaike_project\\2_Data',
        data_set=0,
        data_case=0,
        data_point=0,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=100e3
    )

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6, noise=True,
        f_chop=10e3
    )

    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.1,
        fs_ana=SettingsDATA.fs_resample,
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
        n_order=4, f_filt=[200, 6e3],
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
        no_cluster=3
    )


# --- Setting the src_neuro
class Pipeline:
    """Processing Pipeline for analysing invasive neural activities"""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp0 = DSP(settings.SettingsDSP_LFP)
        self.dsp1 = DSP(settings.SettingsDSP_SPK)
        self.sda = SpikeDetection(settings.SettingsSDA)
        self.fe = FeatureExtraction(settings.SettingsFE)
        self.cl = Clustering(settings.SettingsCL)
        self.signals = PipelineSignal(
            fs_ana=settings.SettingsDATA.fs_resample,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = str()
        self.path2settings = "src_neuro/pipeline_v1.py"

    def generate_folder(self, name: str) -> str:
        if not os.path.exists(self.path2runs):
            os.mkdir(self.path2runs)

        path2figure = os.path.join(self.path2runs, name)
        if not os.path.exists(path2figure):
            os.mkdir(path2figure)

        # --- Copy settings into this folder
        shutil.copy(src=self.path2settings, dst=path2figure)
        self.path2figure = path2figure
        return path2figure

    def prepare_saving(self) -> dict:
        mdict = {"Settings": self.settings,
                 "frames_out": self.signals.frames_align[0],
                 "frames_pos": self.signals.frames_align[1],
                 "frames_id": self.signals.frames_align[2]}
        return mdict

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        u_inn = np.array(self.preamp0.settings.vcm)
        # ---- Analogue Front End Module ----
        self.signals.u_pre, _ = self.preamp0.pre_amp_chopper(uinp, u_inn)
        self.signals.x_adc, _, self.signals.u_quant = self.adc.adc_ideal(self.signals.u_pre)
        # ---- Digital Pre-processing ----
        self.signals.x_lfp = self.dsp0.filter(self.signals.x_adc)
        self.signals.x_spk = self.dsp1.filter(self.signals.x_adc)
        # ---- Spike detection incl. thresholding ----
        self.signals.x_dly = self.sda.time_delay(self.signals.x_spk)
        # self.x_sda = self.sda.sda_neo(self.x_spk)
        self.signals.x_sda, _ = self.sda.sda_smooth(self.sda.sda_neo(self.signals.x_spk))
        self.signals.x_thr = self.sda.thres_blackrock(self.signals.x_sda)
        # self.signals.x_thr = self.sda.thres_blackrock_runtime(self.signals.x_sda)
        (self.signals.frames_orig, self.signals.frames_align) = self.sda.frame_generation(
            self.signals.x_dly, self.signals.x_sda, self.signals.x_thr
        )
        # ---- Feature Extraction  ----
        self.signals.features = self.fe.fe_pca(self.signals.frames_align[0])
        # ---- Clustering | Classification ----
        (self.signals.frames_align[2]) = self.cl.cluster_kmeans(self.signals.features)
        self.signals.spike_ticks = calc_spiketicks(self.signals.frames_align)

    def run_nsp(self) -> None:
        print("NO FURTHER PROCESSING IS INCLUDED")
