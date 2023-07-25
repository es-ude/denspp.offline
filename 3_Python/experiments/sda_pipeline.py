import os, shutil
import numpy as np

from src.pipeline_signals import PipelineSignal
from src.data_call import SettingsDATA
from src.preamp import PreAmp, SettingsAMP
from src.adc.adc_basic import SettingsADC
from src.adc.adc_sar import ADC_SAR as ADC0
from src.dsp import DSP, SettingsDSP
from src.sda import SpikeDetection, SettingsSDA
from src.feature_extraction import FeatureExtraction, SettingsFeature
from src.clustering import Clustering, SettingsCluster
from src.nsp import calc_spiketicks, calc_interval_timing, calc_firing_rate, calc_autocorrelogram

# --- Configuring the pipeline
class Settings:
    """Settings class for handling the pipeline setting"""
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=100,
        gain=40,
        n_filt=1, f_filt=[0.1, 10e3], f_type="band",
        offset=1e-6, noise=True,
        f_chop=10e3
    )

    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.1,
        fs_ana=100,
        fs_dig=20e3, osr=1, Nadc=12
    )

    # --- Digital filtering for ADC output and CIC
    SettingsDSP_SPK = SettingsDSP(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=1, f_filt=[100],
        type='iir', f_type='butter', b_type='high',
        t_dly=0
    )

    SettingsSDA = SettingsSDA(
        fs=SettingsADC.fs_adc, dx_sda=[8],
        mode_align=1,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.1e-3, 0.1e-3],
        t_dly=0.4e-3,
        window_size=7,
        thr_gain=0.5
    )

# --- Setting the pipeline
class Pipeline_Digital(PipelineSignal):
    """Pipeline for processing SDA in digital approaches"""
    def __init__(self, settings: Settings, fs: float):
        PipelineSignal.__init__(self,
            fs_ana=fs,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )

        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp1 = DSP(settings.SettingsDSP_SPK)
        self.sda = SpikeDetection(settings.SettingsSDA)
        self.used_methods = None

    def run_preprocess(self, uin: np.ndarray, mode_sda: int, mode_thr: int) -> None:
        self.u_in = uin
        u_inn = np.array(self.preamp0.settings.vcm)
        # ---- Analogue Front End Module ----
        self.u_pre = self.preamp0.pre_amp(self.u_in, u_inn)
        self.x_adc, _, self.u_quant = self.adc.adc_ideal(self.u_pre)
        # ---- Digital Pre-processing ----
        self.x_spk = self.dsp1.filter(self.x_adc)
        # ---- Spike detection incl. thresholding ----
        self.x_dly = self.sda.time_delay(self.x_spk)
        self.used_methods = self.__do_sda(self.x_spk, mode_sda, mode_thr)
        (self.frames_orig, self.frames_align, self.x_pos) = self.sda.frame_generation(self.x_dly, self.x_sda, self.x_thr)

    def __do_sda(self, xin: np.ndarray, mode_sda: int, mode_thr: int) -> str:
        # --- Performing SDA
        if mode_sda == 0:
            xsda = self.sda.sda_norm(xin)
            text_sda = 'Normal'
        elif mode_sda == 1:
            xsda = self.sda.sda_neo(xin)
            text_sda = 'NEO'
        elif mode_sda == 2:
            xsda = self.sda.sda_mteo(xin)
            text_sda = 'MTEO'
        elif mode_sda == 3:
            xsda = self.sda.sda_aso(xin)
            text_sda = 'ASO'
        elif mode_sda == 4:
            xsda = self.sda.sda_eed(xin, self.fs_adc)
            text_sda = 'EED'
        else:
            xsda = np.zeros(shape=xin.shape)
            text_sda = 'Err'

        xsda, window = self.sda.sda_smooth(xsda)
        self.x_sda = xsda

        # --- Performing Thresholding
        if mode_thr == 0:
            xthr = self.sda.thres_mad(xsda)
            text_thr = 'MAD'
        elif mode_thr == 1:
            xthr = self.sda.thres_blackrock(xsda)
            text_thr = 'RMS_BL'
        elif mode_thr == 2:
            xthr = self.sda.thres_rms(xsda)
            text_thr = 'RMS'
        elif mode_thr == 3:
            xthr = self.sda.thres_ma(xsda)
            text_thr = 'MA'
        elif mode_thr == 4:
            xthr = self.sda.thres_winsorization(xsda)
            text_thr = 'Wins'
        elif mode_thr == 5:
            xthr = self.sda.thres_salvan_golay(xsda)
            text_thr = 'SG'
        else:
            xthr = np.zeros(shape=xsda.shape)
            text_thr = 'Err'

        self.x_thr = xthr
        return text_sda + '+' + text_thr
