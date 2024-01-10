import os, shutil
import numpy as np

from package.pipeline_signals import PipelineSignal
from package.data.data_call_common import SettingsDATA
from package.dsp.sda import SpikeDetection, SettingsSDA
from package.dsp.fex import FeatureExtraction, SettingsFeature
from package.dsp.cluster import Clustering, SettingsCluster
from package.nsp import calc_spiketicks


# --- Configuring the src_decoder
class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        path='C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten',
        data_set=6, data_case=0, data_point=0,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=30e3
    )
    # ch_sel = [34, 55, 95]

    SettingsSDA = SettingsSDA(
        fs=SettingsDATA.fs_resample, dx_sda=[1],
        mode_align=3,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.4e-3, 0.3e-3],
        t_dly=0.3e-3,
        thr_gain=1
    )

    SettingsFE = SettingsFeature(
        no_features=5
    )
    SettingsCL = SettingsCluster(
        no_cluster=3
    )


# --- Setting the src_decoder
class Pipeline(PipelineSignal):
    def __init__(self, settings: Settings):
        PipelineSignal.__init__(self, settings.SettingsDATA.fs_resample, settings.SettingsDATA.fs_resample, 1)

        self.sda = SpikeDetection(settings.SettingsSDA)
        self.fe = FeatureExtraction(settings.SettingsFE)
        self.cl = Clustering(settings.SettingsCL)

        self.__mode_thres = settings.SettingsSDA.mode_thres
        self.__mode_frame = settings.SettingsSDA.mode_align

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None
        self.path2settings = "src_neuro/pipeline_nev.py"

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

    def run(self, wavein: np.ndarray) -> None:
        self.frames_orig = wavein
        self.frames_align = wavein
        # ----- Feature Extraction and Clustering -----
        self.features = self.fe.fe_pca(self.frames_align)
        (self.cluster_id, self.cluster_no, self.sse) = self.cl.cluster_kmeans(self.features)
        self.spike_ticks = calc_spiketicks(self.x_adc, self.x_pos, self.cluster_id)
