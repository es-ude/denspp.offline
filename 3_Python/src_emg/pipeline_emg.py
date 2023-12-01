import os, shutil
import numpy as np

from package.template.pipeline_signals import PipelineSignal
from package.data.data_call_common import SettingsDATA
from package.pre_amp.preamp import PreAmp, SettingsAMP
from package.adc.adc_basic import SettingsADC
from package.adc.adc_sar import ADC_SAR as ADC0
from package.dsp.dsp import DSP, SettingsDSP
from package.dsp.sda import SpikeDetection, SettingsSDA

# --- Configuring the src_neuro
class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        path='C:\HomeOffice\Arbeit\D_EMG\Daten',
        data_set=0,
        data_case=0,
        data_point=1,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=100e3
    )
    # ch_sel = [34, 55, 95]

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
        thr_gain=1
    )

# --- Setting the src_neuro
class Pipeline(PipelineSignal):
    """"""
    def __init__(self, settings: Settings):
        PipelineSignal.__init__(self,
            fs_ana=settings.SettingsDATA.fs_resample,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )

        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp0 = DSP(settings.SettingsDSP_SPK)
        self.sda = SpikeDetection(settings.SettingsSDA)
        self.fe = int()
        self.cl = int()

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None
        self.path2settings = "src_neuro/pipeline_emg.py"

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
        # ---- Analogue Front End Module ----
        self.u_pre, self.u_chp = self.preamp0.pre_amp_chopper(self.u_in, u_inn)
        # self.u_pre = self.preamp0.pre_amp(self.u_in, self.preamp0.settings.vcm)
        # self.u_chp = self.u_pre
        self.x_adc, _, self.u_quant = self.adc.adc_ideal(self.u_pre)
        # ---- Digital Pre-processing ----
        self.x_spk = self.dsp0.filter(self.x_adc)
        # ---- Spike detection incl. thresholding ----
        # self.x_sda = self.sda.sda_neo(self.x_spk)
        self.x_sda = self.sda.sda_aso(self.x_spk)
        self.x_thr = self.sda.thres_blackrock(self.x_sda)
        # self.x_thr = self.sda.thres_blackrock_runtime(self.x_sda)
        (self.frames_orig, self.frames_align, self.x_pos) = self.sda.frame_generation(self.x_dly, self.x_sda, self.x_thr)
