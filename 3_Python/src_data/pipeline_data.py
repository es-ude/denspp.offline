import os
import shutil
import numpy as np

from package.data.pipeline_signals import PipelineSignal
from package.data.data_call import SettingsDATA
from package.pre_amp.preamp import PreAmp, SettingsAMP
from package.adc.adc_basic import SettingsADC
from package.adc.adc_sar import ADC_SAR as ADC0
from package.dsp.sda import SpikeDetection, SettingsSDA


# --- Configuring the src_neuro
class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        path='C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten',
        data_set=7, data_case=0, data_point=0,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=50e3
    )
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=40,
        n_filt=2, f_filt=[200, 4.5e3], f_type="band",
        offset=0e-6, noise=False,
        f_chop=10e3
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed", dvref=0.1,
        fs_ana=SettingsDATA.fs_resample,
        fs_dig=20e3, osr=1, Nadc=12
    )
    SettingsSDA = SettingsSDA(
        fs=SettingsADC.fs_adc, dx_sda=[1],
        mode_align=2,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.4e-3, 0.4e-3],
        t_dly=0.4e-3,
        window_size=7,
        thr_gain=1,
        thr_min_value=100
    )


# --- Setting the src_neuro
class Pipeline(PipelineSignal):
    def __init__(self, settings: Settings):
        PipelineSignal.__init__(self, settings.SettingsDATA.fs_resample, settings.SettingsADC.fs_adc, settings.SettingsADC.osr)

        self.preamp = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.sda = SpikeDetection(settings.SettingsSDA)

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2figure = None
        self.path2settings = "src_data/pipeline_data.py"

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

    def save_settings(self) -> dict:
        mdict = {"fs_adc": self.fs_adc,
                 "v_pre": self.preamp.settings.gain,
                 "f_filt": self.preamp.settings.f_filt,
                 "n_filt": self.preamp.settings.n_filt,
                 "u_lsb": self.adc.settings.lsb,
                 "n_bit": self.adc.settings.Nadc
        }
        return mdict

    def run_input(self, uin: np.ndarray) -> None:
        self.u_in = uin
        u_inn = np.array(self.preamp.settings.vcm)
        # --- Analogue Frontend
        self.u_pre, _ = self.preamp.pre_amp_chopper(self.u_in, u_inn)
        # self.u_pre = self.preamp0.pre_amp(self.u_in, self.preamp0.settings.vcm)
        self.x_adc = self.adc.adc_ideal(self.u_pre)[0]
