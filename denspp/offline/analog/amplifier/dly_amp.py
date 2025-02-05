from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclass
class SettingsDLY:
    """Individual data class to configure the delay amplifier

    Params:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        t_dly:      Delay value for shifting the input [s]
        offset:     Offset voltage of the amplifier [V]
        noise_en:   Enable noise on output [True/False]
        noise_edev: Input voltage noise spectral density [V/sqrt(Hz)]
    """
    vdd:    float
    vss:    float
    fs_ana: float
    # Amplifier characteristics
    t_dly:  float
    offset: float
    # Noise properties
    noise_en: bool
    noise_edev: float


RecommendedSettingsDLY = SettingsDLY(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3,
    t_dly=0.1e-3,
    offset=0e-6,
    noise_en=False,
    noise_edev=100e-9
)


class DlyAmp(CommonAnalogFunctions):
    handler_noise: ProcessNoise
    settings: SettingsDLY
    __print_device = "delay amplifier"

    def __init__(self, settings_dly: SettingsDLY, settings_noise: SettingsNoise=RecommendedSettingsNoise) -> None:
        """Class for emulating an analogue delay amplifier
        :param settings_dly:        Dataclass for handling the delay amplifier
        :param settings_noise:      Dataclass for handling the noise simulation
        """
        super().__init__(settings_dly)
        self.handler_noise = ProcessNoise(settings_noise, settings_dly.fs_ana)
        self.settings = settings_dly

    @property
    def vcm(self) -> float:
        return (self.settings.vdd + self.settings.vss) / 2

    @property
    def num_dly_taps(self) -> int:
        return int(self.settings.fs_ana * self.settings.t_dly)

    def do_simple_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a simple delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros(u_inp.shape) + self.vcm
        uout[self.num_dly_taps:] = u_inp[:-self.num_dly_taps]
        return self.voltage_clipping(uout)

    def do_recursive_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a recursive delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros(u_inp.shape)
        uout[:self.num_dly_taps] = u_inp[-self.num_dly_taps:]
        uout[self.num_dly_taps:] = u_inp[:-self.num_dly_taps]
        return self.voltage_clipping(uout)

    def do_allpass_first_order(self, uin: np.ndarray, f_b: float=1.0) -> np.ndarray:
        """Performing a 1st order all-pass filter (IIR) for adding time delay
        Args:
            uin:    Input voltage [V]
            f_b:    Frequency of used delay [Hz]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * f_b / self.settings.fs_ana)
        iir_c0 = (val - 1) / (val + 1)

        b = [iir_c0, 1.0]
        a = [1.0, iir_c0]
        return self.voltage_clipping(lfilter(b, a, uin))

    def do_allpass_second_order(self, uin: np.ndarray, f_b: float=1.0, bandwidth: float=0.5) -> np.ndarray:
        """Performing a 2nd order all-pass filter (IIR) for adding time delay
        Args:
            uin:        Input voltage [V]
            f_b:        Frequency of used delay [Hz]
            bandwidth:  Bandwidth frequency [Hz]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * bandwidth / self.settings.fs_ana)
        iir_c0 = (val - 1) / (val + 1)
        iir_c1 = -np.cos(2 * np.pi * f_b / self.settings.fs_ana)

        b = [-iir_c0, iir_c1*(1-iir_c0), 1.0]
        a = [1.0, iir_c1*(1-iir_c0), -iir_c0]
        return self.voltage_clipping(lfilter(b, a, uin))
