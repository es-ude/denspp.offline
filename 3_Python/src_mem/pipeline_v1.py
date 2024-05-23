from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD, PipelineSignal
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_sar import SettingsADC, ADC_SAR


# --- Configuring the src_neuro
class SettingsPipe:
    """Settings class for setting-up the pipeline"""
    def __init__(self, fs_ana: float):
        self.SettingsAMP.fs_ana = fs_ana
        self.SettingsADC.fs_ana = fs_ana

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=0.0,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6, noise_en=True,
        f_chop=10e3
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.1,
        fs_ana=0.0,
        fs_dig=20e3, osr=1, Nadc=12
    )


class Pipeline(PipelineCMD):
    """Processing Pipeline for analysing invasive neural activities"""
    def __init__(self, fs_ana: float):
        super().__init__()
        self._path2pipe = abspath(__file__)
        self.generate_folder('runs', '_mem')

        settings = SettingsPipe(fs_ana)
        self.signals = PipelineSignal()
        self.signals.fs_ana = settings.SettingsADC.fs_ana
        self.signals.fs_adc = settings.SettingsADC.fs_adc
        self.signals.fs_dig = settings.SettingsADC.fs_dig

        self.__preamp0 = PreAmp(settings.SettingsAMP)
        self.__adc = ADC_SAR(settings.SettingsADC)

    def prepare_saving(self) -> dict:
        """"""
        mdict = {"fs_ana": self.signals.fs_ana,
                 "fs_adc": self.signals.fs_adc,
                 "fs_dig": self.signals.fs_dig,
                 "u_in": self.signals.u_in,
                 "x_adc": self.signals.x_adc}
        return mdict

    def do_plotting(self, data: PipelineSignal, channel: int) -> None:
        """Function to plot results from spike sorting"""
        import package.plot.plot_pipeline as plt_neuro
        path2save = self.path2save

    def run(self, uinp: np.ndarray) -> None:
        self.signals.u_in = uinp
        # ---- Analogue Front End Module ----
        self.signals.u_pre = self.__preamp0.pre_amp_chopper(uinp, np.array(self.__preamp0.vcm))[0]
        self.signals.x_adc = self.__adc.adc_ideal(self.signals.u_pre)[0]
