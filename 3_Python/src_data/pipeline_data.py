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
        data_set=1, data_case=0, data_point=0,
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
class Pipeline:
    def __init__(self, settings: Settings):
        self.signals = PipelineSignal(
            fs_ana=settings.SettingsDATA.fs_resample,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )
        self.__preamp = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__sda = SpikeDetection(settings.SettingsSDA)

        self.path2logs = "logs"
        self.path2runs = "runs"
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
        mdict = {"fs_adc": self.__adc.settings.fs_adc,
                 "v_pre": self.__preamp.settings.gain,
                 "f_filt": self.__preamp.settings.f_filt,
                 "n_filt": self.__preamp.settings.n_filt,
                 "u_lsb": self.__adc.settings.lsb,
                 "n_bit": self.__adc.settings.Nadc
        }
        return mdict

    def run_input(self, uin: np.ndarray, spike_xpos: np.ndarray, spike_xoffset: int) -> None:
        self.signals.u_in = uin
        u_inn = np.array(self.__preamp.settings.vcm)
        # --- Analogue Frontend
        self.signals.u_pre, _ = self.__preamp.pre_amp_chopper(self.signals.u_in, u_inn)
        # self.u_pre = self.preamp0.pre_amp(self.u_in, self.preamp0.settings.vcm)
        self.signals.x_adc = self.__adc.adc_ideal(self.signals.u_pre)[0]

        self.signals.frames_align, self.signals.x_pos = self.__sda.frame_generation_pos(
            self.signals.x_adc, spike_xpos, spike_xoffset
        )[1:]
