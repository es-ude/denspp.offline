import dataclasses
import numpy as np
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsCUR:
    """Individual data class to configure the current amplifier

    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        gain:       Amplification [V/V]
        offset_v:   Offset voltage of current amplifier [V]
        offset_i:   Offset current of current amplifier [A]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
    """
    vdd:        float
    vss:        float
    fs_ana:     float
    # Amplifier characteristics
    gain:       int
    offset_v:   float
    offset_i:   float
    # Settings for parasitic
    noise_en:   bool
    para_en:    bool

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsCUR = SettingsCUR(
    vdd=0.6, vss=0.6,
    fs_ana=50e3,
    gain=40,
    offset_v=1e-6, offset_i=1e-12,
    noise_en=False,
    para_en=False
)


class CurrentAmplifier(ProcessNoise):
    """Class for emulating an analogue current amplifier"""
    _settings_noise: SettingsNoise
    __print_device = "current amplifier"

    def __init__(self, settings_dev: SettingsCUR, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self._settings.vdd] = self._settings.vdd
        uin[uin < self._settings.vss] = self._settings.vss
        return uin

    def pre_amp(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs the pre-amplification (single, normal) with input signal

        Args:
            uinp:   Positive input voltage [V]
            uinn:   Negative input voltage [V]

        Returns:
            Test signal
        """
        du = uinp - uinn
        u_out = du
        u_out += self._settings.gain * self._settings.offset_v
        u_out += self._settings.vcm

        # Adding noise
        if self._settings.noise_en:
            u_out += self._settings.gain * self._gen_noise_real(du.size)

        return self.__voltage_clipping(u_out)
