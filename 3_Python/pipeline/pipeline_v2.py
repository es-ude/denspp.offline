import os, shutil
import numpy as np

from src.pipeline_signals import PipelineSignal
from src.data_call import SettingsDATA
from src.preamp import PreAmp, SettingsAMP
from src.adc.adc_basic import SettingsADC
from src.adc.adc_sar import ADC_SAR as ADC0
from src.dsp import DSP, SettingsDSP
from src.sda import SDA, SettingsSDA
from src.feature_extraction import FeatureExtraction, SettingsFeature
from src.clustering import Clustering, SettingsCluster
from src.nsp import calc_spiketicks, calc_interval_timing

# --- Configuring the pipeline
class Settings:
    """Settings class for handling the pipeline setting"""
    SettingsDATA = SettingsDATA(
        path="C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten",
        data_set=1, data_point=0,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=50e3
    )
    # ch_sel = [34, 55, 95]

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=0, noise=0
    )

    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        osr=1, Nadc=12,
        dvref=0.1,
        fs_ana=SettingsDATA.fs_resample, fs_adc=20e3
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
        n_order=2, f_filt=[100, 6e3],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
    )

    SettingsSDA = SettingsSDA(
        fs=SettingsADC.fs_adc, dx_sda=[1],
        mode_thres=2, mode_align=3,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.4e-3, 0.3e-3],
        t_dly=0.3e-3
    )

    SettingsFE = SettingsFeature(
        no_features=3
    )
    SettingsCL = SettingsCluster(
        no_cluster=3
    )
class Pipeline(PipelineSignal):
    def __init__(self, settings: Settings):
        PipelineSignal.__init__(self, settings.SettingsDATA.fs_resample, settings.SettingsADC.fs_adc)

        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp0 = DSP(settings.SettingsDSP_LFP)
        self.dsp1 = DSP(settings.SettingsDSP_SPK)
        self.sda = SDA(settings.SettingsSDA)
        self.fe = FeatureExtraction(settings.SettingsFE)
        self.cl = Clustering(settings.SettingsCL)
        self.dae = None

        self.__mode_thres = settings.SettingsSDA.mode_thres
        self.__mode_frame = settings.SettingsSDA.mode_align

        self.path2model = "models"
        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None
        self.path2settings = "pipeline/pipeline_v2.py"

    def saving_results(self, name: str) -> str:
        if not os.path.exists(self.path2runs):
            os.mkdir(self.path2runs)

        path2figure = os.path.join(self.path2runs, name)
        if not os.path.exists(path2figure):
            os.mkdir(path2figure)

        # --- Copy settings into this folder
        shutil.copy(src=self.path2settings, dst=path2figure)
        self.path2figure = path2figure

        return path2figure

    def run(self, uin: np.ndarray) -> None:
        self.u_in = uin
        u_inn = np.array(self.preamp0.settings.vcm)
        # ----- Analogue Front End Module  -----
        (self.u_pre) = self.preamp0.pre_amp(self.u_in, u_inn)
        self.x_adc = self.adc.adc_ideal(self.u_pre)[0]
        # --- Digital Pre-processing ----
        self.x_lfp = self.dsp0.filter(self.x_adc)
        self.x_spk = self.dsp1.filter(self.x_adc)
        # --- Spike detection incl. thresholding
        self.x_dly = self.sda.time_delay(self.x_spk)
        (self.x_sda, self.x_thr) = self.sda.sda(self.x_spk, self.__mode_thres)
        (self.frames_orig, self.x_pos) = self.sda.frame_generation(self.x_spk, self.x_sda, self.x_thr)
        self.frames_align = self.sda.frame_aligning(self.frames_orig, self.__mode_frame)
        # --- Adding denoising
        val_max = 48
        self.frames_denoised = self.dae.predict(self.frames_align/val_max)
        self.frames_denoised = self.frames_denoised * val_max
        # --- Feature Extraction ---
        self.features = self.fe.fe_pca(self.frames_denoised)
        # --- Classification ---
        (self.cluster_id, self.cluster_no, self.sse) = self.cl.cluster_kmeans(self.features)
        self.spike_ticks = calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id)
        self.its = calc_interval_timing(self.spike_ticks, self.fs_dig)
