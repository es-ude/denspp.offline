from os.path import abspath
import numpy as np

from package.pipeline_signals import PipelineSignal
from package.pipeline_cmds import PipelineCMD
from package.data_call.call_handler import SettingsDATA
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_sar import SettingsADC, ADC_SAR as ADC0


# --- Configuring the src_neuro
class SettingsPipe:
    """Settings class for handling the src_neuro setting"""
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6, noise_en=True,
        f_chop=10e3
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.1,
        fs_ana=SettingsDATA.fs_resample,
        fs_dig=20e3, osr=1, Nadc=12
    )


class Pipeline(PipelineCMD):
    """Processing Pipeline for analysing invasive neural activities"""
    def __init__(self, settings: SettingsPipe):
        super().__init__()
        self.settings = settings
        self.signals = PipelineSignal(
            fs_ana=settings.SettingsDATA.fs_resample,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )

        self.__preamp0 = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)

        self.path2pipe = abspath(__file__)
        self.generate_folder('runs', '_mem')

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        # ---- Analogue Front End Module ----
        self.signals.u_pre, _ = self.__preamp0.pre_amp_chopper(uinp, np.array(self.__preamp0.vcm))
        self.signals.x_adc, _, self.signals.u_quant = self.__adc.adc_ideal(self.signals.u_pre)
