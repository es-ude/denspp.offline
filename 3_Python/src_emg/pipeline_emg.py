from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD
from package.pipeline_signals import PipelineSignal
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_basic import SettingsADC
from package.analog.adc_sar import ADC_SAR as ADC0
from package.digital.dsp import DSP, SettingsDSP
from package.digital.sda import SpikeDetection, SettingsSDA


def get_envelope(signal: np.ndarray, size_envelope: int) -> np.ndarray:
    """"""
    out = np.zeros(signal.shape)
    for idx in range(signal.size-size_envelope):
        xdata = signal[idx:idx+size_envelope]
        data_max = xdata.max()
        out[idx:idx+size_envelope] = data_max

    return out


class SettingsPipe:
    """Settings class for setting-up the pipeline"""
    def __init__(self, fs_ana: float):
        self.SettingsAMP.fs_ana = fs_ana
        self.SettingsADC.fs_ana = fs_ana

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=0.0,
        gain=40,
        n_filt=2, f_filt=[0.1, 450], f_type="band",
        offset=1e-6,
        f_chop=10e3,
        noise_en=False,
        noise_edev=100e-9
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.5,
        fs_ana=0.0,
        fs_dig=1e3, osr=1, Nadc=16
    )
    SettingsDSP_SPK = SettingsDSP(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=2, f_filt=[20, 450],
        type='iir', f_type='butter', b_type='bandpass',
        t_dly=0
    )


class Pipeline(PipelineCMD):
    """"""
    def __init__(self, fs_ana: float):
        super().__init__()
        self._path2pipe = abspath(__file__)
        self.generate_folder('runs', '_emg')

        settings = SettingsPipe(fs_ana)
        self.signal = PipelineSignal()
        self.signal.fs_ana = settings.SettingsADC.fs_ana
        self.signal.fs_adc = settings.SettingsADC.fs_adc
        self.signal.fs_dig = settings.SettingsADC.fs_dig

        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp0 = DSP(settings.SettingsDSP_SPK)

    def prepare_saving(self) -> dict:
        """"""
        mdict = {
            "fs_ana": self.signal.fs_ana,
            "fs_adc": self.signal.fs_adc,
            "fs_dig": self.signal.fs_dig,
            "u_in": self.signal.u_in,
            "x_adc": self.signal.x_adc,
            "x_flt": self.signal.x_spk,
            "x_env": self.signal.x_sda
        }
        return mdict

    def run(self, u_inp: np.ndarray) -> None:
        self.signal.u_in = u_inp
        u_inn = np.array(self.preamp0.vcm)
        # ---- Analogue Front End Module ----
        u_pre = self.preamp0.pre_amp(u_inp, u_inn)
        self.signal.x_adc = self.adc.adc_ideal(u_pre)[0]
        # ---- Digital Pre-processing ----
        self.signal.x_spk = self.dsp0.filter(self.signal.x_adc)
        x0 = np.abs(self.signal.x_spk)
        self.signal.x_sda = get_envelope(x0, 200)
