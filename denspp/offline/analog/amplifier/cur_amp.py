from dataclasses import dataclass
import numpy as np
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, DefaultSettingsNoise


@dataclass
class SettingsCUR:
    """Individual data class to configure the current amplifier
    Attributes:
        vdd:            Positive supply voltage [V]
        vss:            Negative supply voltage [V]
        fs_ana:         Sampling frequency of input [Hz]
        transimpedance: Transimpedance value [V/A]
        offset_v:       Offset voltage of current amplifier [V]
        offset_i:       Offset current of current amplifier [A]
        noise_en:       Enable noise on output [True / False]
        para_en:        Enable parasitic [True / False]
    """
    vdd:            float
    vss:            float
    fs_ana:         float
    # Amplifier characteristics
    transimpedance: float
    offset_v:       float
    offset_i:       float
    # Settings for parasitic
    noise_en:       bool
    para_en:        bool

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


DefaultSettingsCUR = SettingsCUR(
    vdd=0.9, vss=-0.9,
    fs_ana=50e3,
    transimpedance=1e3,
    offset_v=1e-6, offset_i=1e-12,
    noise_en=False,
    para_en=False
)


class CurrentAmplifier(CommonAnalogFunctions):
    _handler_noise: ProcessNoise
    _settings: SettingsCUR
    __print_device = "current amplifier"

    def __init__(self, settings_dev: SettingsCUR, settings_noise: SettingsNoise=DefaultSettingsNoise) -> None:
        """Class for emulating an analogue current amplifier
        :param settings_dev:        Dataclass for handling the current amplifier
        :param settings_noise:      Dataclass for handling the noise and parasitic simulation
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._handler_noise = ProcessNoise(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def _add_parasitic(self, size: int, resistance: float=1.0) -> np.ndarray:
        u_para = np.zeros((size, ))
        u_para += self._settings.transimpedance * self._settings.offset_i
        u_para += self._settings.offset_v
        u_para += self._settings.vcm
        if self._settings.noise_en:
            u_para += self._handler_noise.gen_noise_real_volt(size, resistance)

        return u_para

    def _build_ref_difference(self, u_ref: np.ndarray | float) -> np.ndarray | float:
        return u_ref - self._settings.vcm

    def transimpedance_amplifier(self, i_in: np.ndarray, u_ref: np.ndarray | float) -> np.ndarray:
        """Performing the transimpedance amplifier (single, normal) with input signal
        Args:
            i_in:    Input current [A]
            u_ref:   Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = self._settings.transimpedance * i_in
        u_out += self._build_ref_difference(u_ref)
        u_out += self._add_parasitic(u_out.size)
        return self.clamp_voltage(u_out)

    def instrumentation_amplifier(self, i_in: np.ndarray, u_off: np.ndarray | float, v_gain: float=1.0) -> np.ndarray:
        """Using an instrumentation amplifier for current sensing
        Args:
            i_in:    Input current [A]
            u_off:   Offset output voltage [V]
            v_gain: Gain of Amplifier [V/V]
        Returns:
            Corresponding numpy array with output voltage
        """
        r_sense = self._settings.transimpedance
        u_out = r_sense * i_in
        u_out += self._build_ref_difference(u_off)
        u_out += self._add_parasitic(u_out.size, r_sense)
        return self.clamp_voltage(v_gain * u_out)

    def push_amplifier(self, i_in: np.ndarray, u_ref: float) -> np.ndarray:
        """Performing the CMOS push/source current amplifier
        Args:
            i_in:   Input current [A]
            u_ref:   Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = np.zeros_like(i_in)
        x_neg = np.argwhere(i_in < u_ref)
        u_out[x_neg,] = i_in[x_neg,] * self._settings.transimpedance
        u_out += self._add_parasitic(u_out.size, self._settings.transimpedance)
        return self.clamp_voltage(u_out)

    def pull_amplifier(self, i_in: np.ndarray, u_ref: float) -> np.ndarray:
        """Performing the CMOS pull/sink current amplifier
        Args:
            i_in:   Input current [A]
            u_ref:  Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = np.zeros_like(i_in)
        x_pos = np.argwhere(i_in >= u_ref)
        u_out[x_pos, ] = i_in[x_pos,] * self._settings.transimpedance
        u_out += self._add_parasitic(u_out.size, self._settings.transimpedance)
        return self.clamp_voltage(u_out)

    def push_pull_amplifier(self, i_in: np.ndarray, u_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """Performing the CMOS push-pull current amplifier
        Args:
            i_in: Input current [A]
        Returns:
            Tuple with corresponding numpy array of output voltage (positive, negative)
        """
        u_pos = self.pull_amplifier(i_in, u_ref)
        u_neg = self.push_amplifier(i_in, u_ref)
        return u_pos, u_neg

    def push_pull_abs_amplifier(self, i_in: np.ndarray, u_ref: float) -> np.ndarray:
        """Performing the CMOS push-pull current absolute amplifier
        Args:
            i_in:   Input current [A]
            u_ref:  Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = self.pull_amplifier(i_in, u_ref)
        u_out -= self.push_amplifier(i_in, u_ref)
        return u_out
