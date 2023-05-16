import os, shutil
import numpy as np

from src.pipeline_signals import PipelineSignal
from src.data_call import SettingsDATA
from src.sda import SDA, SettingsSDA
from src.feature_extraction import FeatureExtraction, SettingsFeature
from src.clustering import Clustering, SettingsCluster
from src.nsp import calc_spiketicks, calc_interval_timing

# --- Configuring the pipeline
class Settings:
    """Settings class for handling the pipeline setting"""
    SettingsDATA = SettingsDATA(
        path='C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten',
        data_set=1, data_point=0,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=30e3
    )
    # ch_sel = [34, 55, 95]

    SettingsSDA = SettingsSDA(
        fs=SettingsDATA.fs_resample, dx_sda=[1],
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

# --- Setting the pipeline
class Pipeline(PipelineSignal):
    def __init__(self, settings: Settings):
        PipelineSignal.__init__(self, settings.SettingsDATA.fs_resample, settings.SettingsDATA.fs_resample)

        self.sda = SDA(settings.SettingsSDA)
        self.fe = FeatureExtraction(settings.SettingsFE)
        self.cl = Clustering(settings.SettingsCL)

        self.__mode_thres = settings.SettingsSDA.mode_thres
        self.__mode_frame = settings.SettingsSDA.mode_align

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None

    def saving_results(self, name: str) -> str:
        if not os.path.exists(self.path2runs):
            os.mkdir(self.path2runs)

        path2figure = os.path.join(self.path2runs, name)
        if not os.path.exists(path2figure):
            os.mkdir(path2figure)

        # --- Copy settings into this folder
        shutil.copy(src="pipeline/pipeline_nev.py", dst=path2figure)
        self.path2figure = path2figure

        return path2figure

    def run(self, uin: np.ndarray) -> None:
        self.u_in = uin
        # ---- Spike Detection ----
        (self.x_sda, self.x_thr) = self.sda.sda(self.x_spk, self.__mode_thres)
        # self.x_sda = self.dsp0.dig_filt_iir(self.x_sda)
        (self.frames_orig, self.x_pos) = self.sda.frame_generation(self.x_dly, self.x_sda, self.x_thr)
        self.frames_align = self.sda.frame_aligning(self.frames_orig, self.__mode_frame)
        # ----- Feature Extraction -----
        self.features = self.fe.fe_pca(self.frames_align)
        (self.cluster_id, self.cluster_no, self.sse) = self.cl.cluster_kmeans(self.features)
        self.spike_ticks = self.spiketicks(self.x_adc, self.x_pos, self.cluster_id)
