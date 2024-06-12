from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD, PipelineSignal
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_basic import SettingsADC
from package.analog.adc_sar import ADC_SAR as ADC0
from package.digital.sda import SpikeDetection, SettingsSDA


# --- Configuring the src_neuro
class _SettingsPipe:
    """Settings class for setting-up the pipeline"""
    def __init__(self, fs_ana: float):
        self.SettingsAMP.fs_ana = fs_ana
        self.SettingsADC.fs_ana = fs_ana

    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=0.0,
        gain=100,
        n_filt=2, f_filt=[200, 8e3], f_type="band",
        offset=0e-6, noise_en=False,
        f_chop=20e3,
        noise_edev=100e-9
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed", dvref=0.1,
        fs_ana=0.0,
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
class Pipeline(PipelineCMD):
    def __init__(self, fs_ana: float):
        super().__init__()
        self._path2pipe = abspath(__file__)
        # self.generate_folder('runs', '_data')

        settings = _SettingsPipe(fs_ana)
        self.fs_ana = fs_ana
        self.fs_dig = settings.SettingsADC.fs_dig
        self.fs_adc = settings.SettingsADC.fs_dig
        self.lsb = settings.SettingsADC.lsb
        self.clean_pipeline()

        self.__preamp = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__sda = SpikeDetection(settings.SettingsSDA)

        self.frame_left_windowsize = self.__sda.frame_start + int(self.__sda.offset_frame / 2)
        self.frame_right_windowsize = self.__sda.frame_ends + int(self.__sda.offset_frame / 2)

    def clean_pipeline(self) -> None:
        """Cleaning the signals of the pipeline"""
        self.signals = PipelineSignal()
        self.signals.fs_ana = self.fs_ana
        self.signals.fs_adc = self.fs_adc
        self.signals.fs_dig = self.fs_dig

    def prepare_saving(self) -> dict:
        """Getting processing data of selected signals"""
        mdict = {"fs_adc": self.signals.fs_adc}
        return mdict

    def do_plotting(self, data: PipelineSignal, channel: int) -> None:
        """Function to plot results"""
        pass

    def run_input(self, uin: np.ndarray, spike_xpos: np.ndarray, spike_xoffset=0) -> None:
        """Processing the raw data for frame generation
        Args:
            uin: Array of the 1D-transient signal
            spike_xpos: List of all spike positions from groundtruth
            spike_xoffset: Time delay between spike_xpos and real spike
        """
        self.run_minimal(uin)

        self.signals.frames_align, self.signals.x_pos = self.__sda.frame_generation_pos(
            self.signals.x_adc, spike_xpos, spike_xoffset
        )[1:]

    def run_minimal(self, uin: np.ndarray) -> None:
        """Processing the input for getting the reshaped digital datastream
        Args:
            uin: Array of the 1D-transient signal
        """
        self.signals.u_in = uin
        u_inn = np.array(self.__preamp.vcm)
        # --- Analogue Frontend
        # self.signals.u_pre, _ = self.__preamp.pre_amp_chopper(self.signals.u_in, u_inn)
        self.signals.u_pre = self.__preamp.pre_amp(self.signals.u_in, u_inn)
        self.signals.x_adc = self.__adc.adc_ideal(self.signals.u_pre)[0]
