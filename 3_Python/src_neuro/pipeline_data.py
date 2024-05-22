from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD
from package.pipeline_signals import PipelineSignal
from package.data_call.call_handler import SettingsDATA
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_basic import SettingsADC
from package.analog.adc_sar import ADC_SAR as ADC0
from package.digital.sda import SpikeDetection, SettingsSDA


# --- Configuring the src_neuro
class Settings:
    """Settings class for handling the src_neuro setting"""
    SettingsDATA = SettingsDATA(
        # path='../2_Data',
        # path='/media/erbsloeh/ExtremeSSD/0_Invasive',
        path='C:/HomeOffice/Data_Neurosignal',
        data_set=1, data_case=0, data_point=0,
        t_range=[0],
        ch_sel=[],
        fs_resample=50e3
    )
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=SettingsDATA.fs_resample,
        gain=100,
        n_filt=2, f_filt=[200, 8e3], f_type="band",
        offset=0e-6, noise_en=False,
        f_chop=20e3,
        noise_edev=100e-9
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
class Pipeline(PipelineCMD):
    def __init__(self, settings: Settings):
        super().__init__()
        self.path2pipe = abspath(__file__)
        # self.generate_folder('runs', '_data')

        self.signals = PipelineSignal(
            fs_ana=settings.SettingsDATA.fs_resample,
            fs_adc=settings.SettingsADC.fs_adc,
            osr=settings.SettingsADC.osr
        )
        self.__preamp = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__sda = SpikeDetection(settings.SettingsSDA)

        self.frame_left_windowsize = self.__sda.frame_start + int(self.__sda.offset_frame / 2)
        self.frame_right_windowsize = self.__sda.frame_ends + int(self.__sda.offset_frame / 2)

        self.path2logs = "logs"
        self.path2runs = "runs"
        self.path2settings = "src_data/pipeline_data.py"

    def save_settings(self) -> dict:
        mdict = {"fs_adc": self.__adc.settings.fs_adc,
                 "v_pre": self.__preamp._settings_dev.gain,
                 "f_filt": self.__preamp._settings_dev.f_filt,
                 "n_filt": self.__preamp._settings_dev.n_filt,
                 "u_lsb": self.__adc.settings.lsb,
                 "n_bit": self.__adc.settings.Nadc
        }
        return mdict

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
