import dataclasses
import numpy as np
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsCUR:
    """Individual data class to configure the current amplifier

    Params:
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


RecommendedSettingsCUR = SettingsCUR(
    vdd=0.9, vss=-0.9,
    fs_ana=50e3,
    transimpedance=1e3,
    offset_v=1e-6, offset_i=1e-12,
    noise_en=False,
    para_en=False
)


class CurrentAmplifier(CommonAnalogFunctions):
    handler_noise: ProcessNoise
    _settings: SettingsCUR
    __print_device = "current amplifier"

    def __init__(self, settings_dev: SettingsCUR, settings_noise: SettingsNoise=RecommendedSettingsNoise) -> None:
        """Class for emulating an analogue current amplifier
        :param settings_dev:        Dataclass for handling the current amplifier
        :param settings_noise:      Dataclass for handling the noise and parasitic simulation
        """
        super().__init__(settings_dev)
        self.handler_noise = ProcessNoise(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    @property
    def vcm(self) -> float:
        return (self._settings.vdd + self._settings.vss) / 2

    def __add_parasitic(self, size: int, resistance=1.0) -> np.ndarray:
        """"""
        u_para = np.zeros((size, ))
        u_para += self._settings.transimpedance * self._settings.offset_i
        u_para += self._settings.offset_v
        u_para += self.vcm
        # Adding noise
        if self._settings.noise_en:
            u_para += self.handler_noise.gen_noise_real_volt(size, resistance)

        return u_para

    def transimpedance_amplifier(self, iin: np.ndarray, uref: np.ndarray | float) -> np.ndarray:
        """Performing the transimpedance amplifier (single, normal) with input signal
        Args:
            iin:    Input current [A]
            uref:   Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = self._settings.transimpedance * iin + uref
        u_out += self.__add_parasitic(u_out.size)
        return self.voltage_clipping(u_out)

    def instrumentation_amplifier(self, iin: np.ndarray, uoff: np.ndarray | float, v_gain=1.0) -> np.ndarray:
        """Using an instrumentation amplifier for current sensing
        Args:
            iin:    Input current [A]
            uoff:   Offset output voltage [V]
            v_gain: Gain of Amplifier [V/V]
        Returns:
            Corresponding numpy array with output voltage
        """
        r_sense = self._settings.transimpedance / v_gain
        u_out = r_sense * iin + uoff
        u_out += self.__add_parasitic(u_out.size, r_sense)
        return u_out

    def push_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS push/source current amplifier
        Args:
            iin: Input current [A]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = np.zeros(iin.shape)
        x_neg = np.argwhere(iin < 0)
        u_out[x_neg,] = iin[x_neg,] * self._settings.transimpedance
        u_out += self.__add_parasitic(u_out.size, self._settings.transimpedance)
        return self.voltage_clipping(u_out)

    def pull_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS pull/sink current amplifier
        Args:
            iin: Input current [A]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = np.zeros(iin.shape)
        x_pos = np.argwhere(iin >= 0)
        u_out[x_pos, ] = iin[x_pos, ] * self._settings.transimpedance
        u_out += self.__add_parasitic(u_out.size, self._settings.transimpedance)
        return self.voltage_clipping(u_out)

    def push_pull_amplifier(self, iin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performing the CMOS push-pull current amplifier
        Args:
            iin: Input current [A]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_pos = self.pull_amplifier(iin)
        u_neg = self.push_amplifier(iin)
        return u_pos, u_neg

    def push_pull_abs_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS push-pull current absolute amplifier
        Args:
            iin: Input current [A]
        Returns:
            Corresponding numpy array with output voltage
        """
        u_out = self.pull_amplifier(iin)
        u_out -= self.push_amplifier(iin)
        return u_out
