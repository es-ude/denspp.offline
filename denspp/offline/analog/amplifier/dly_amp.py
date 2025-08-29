from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, DefaultSettingsNoise


@dataclass
class SettingsDLY:
    """Individual data class to configure the delay amplifier
    Attributes:
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
    t_dly:  float
    offset: float
    noise_en: bool
    noise_edev: float

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2

    @property
    def f_break_norm(self) -> float:
        return 1 / (self.fs_ana * self.t_dly)

    @property
    def num_dly_taps(self) -> int:
        val = int(self.fs_ana * self.t_dly)
        if val <= 0:
            raise ValueError("t_dly and fs_ana must be positive")
        return val


DefaultSettingsDLY = SettingsDLY(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3,
    t_dly=0.5e-3,
    offset=0e-6,
    noise_en=False,
    noise_edev=100e-9
)


class DelayAmplifier(CommonAnalogFunctions):
    _handler_noise: ProcessNoise
    _settings: SettingsDLY

    def __init__(self, settings_dev: SettingsDLY, settings_noise: SettingsNoise=DefaultSettingsNoise) -> None:
        """Class for emulating an analogue delay amplifier
        (More infos on: https://thewolfsound.com/allpass-filter/)
        :param settings_dev:        Dataclass for handling the delay amplifier
        :param settings_noise:      Dataclass for handling the noise simulation
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._handler_noise = ProcessNoise(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def do_simple_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a simple delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros_like(u_inp) + self._settings.vcm
        uout[self._settings.num_dly_taps:] = u_inp[:-self._settings.num_dly_taps]
        return self.clamp_voltage(uout)

    def do_recursive_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a recursive delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros(u_inp.shape)
        uout[:self._settings.num_dly_taps+1] = u_inp[-self._settings.num_dly_taps-1:]
        uout[self._settings.num_dly_taps:] = u_inp[:-self._settings.num_dly_taps]
        return self.clamp_voltage(uout)

    def do_allpass_first_order(self, u_in: np.ndarray) -> np.ndarray:
        """Performing a 1st order all-pass filter (IIR) for adding time delay (1/t_dly = f_break)
        Args:
            u_in:    Input voltage [V]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * self._settings.f_break_norm)
        iir_c0 = (val - 1) / (val + 1)

        b = [iir_c0, 1.0]
        a = [1.0, iir_c0]
        return self.clamp_voltage(lfilter(b, a, u_in))

    def do_allpass_second_order(self, u_in: np.ndarray, bandwidth: float) -> np.ndarray:
        """Performing a 2nd order all-pass filter (IIR) for adding time delay (1/t_dly = f_break)
        Args:
            u_in:       Input voltage [V]
            bandwidth:  Bandwidth frequency [Hz]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * bandwidth / self._settings.fs_ana)
        iir_c0 = (val - 1) / (val + 1)
        iir_c1 = -np.cos(2 * np.pi * self._settings.f_break_norm)

        b = [-iir_c0, iir_c1*(1-iir_c0), 1.0]
        a = [1.0, iir_c1*(1-iir_c0), -iir_c0]
        return self.clamp_voltage(lfilter(b, a, u_in))
