import os, shutil
import numpy as np

from package.pipeline_signals import PipelineSignal
from package.data_call.call_handler import SettingsDATA
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_basic import SettingsADC
from package.analog.adc_sar import ADC_SAR as ADC0
from package.digital.dsp import DSP, SettingsDSP
from package.digital.sda import SpikeDetection, SettingsSDA


class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        path='C:/HomeOffice/Data_EMG',
        data_set=0,
        data_case=0,
        data_point=1,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=2e3
    )
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=40,
        n_filt=2, f_filt=[0.1, 0.8e3], f_type="band",
        offset=1e-6, noise=True,
        f_chop=10e3
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.5,
        fs_ana=SettingsDATA.fs_resample,
        fs_dig=1e3, osr=1, Nadc=16
    )
    SettingsDSP_SPK = SettingsDSP(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=4, f_filt=[0.1, 0.45e3],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
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
        self.u_pre = self.preamp0.pre_amp(self.u_in, u_inn)
        self.x_adc = self.adc.adc_ideal(self.u_pre)[0]
        # ---- Digital Pre-processing ----
        self.x_spk = self.dsp0.filter(self.x_adc)
