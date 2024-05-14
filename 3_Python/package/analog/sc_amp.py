import dataclasses
import numpy as np
from scipy.signal import square
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsSC:
    """Individual data class to configure switched-capacitor circuits

    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        fs_sc:      Switching frequency of the modulator [Hz]
        gain:       Amplification [V/V]
        offset:     Offset voltage of the amplifier [V]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
    """
    vdd:        float
    vss:        float
    fs_ana:     float
    fs_sc:      float
    # Amplifier characteristics
    gain:       int
    offset:     float
    noise_en:   bool
    para_en:    bool

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsAMP = SettingsSC(
    vdd=0.6, vss=0.6,
    fs_ana=50e3, fs_sc=10e3,
    gain=40,
    offset=0e-6,
    noise_en=False,
    para_en=False
)


class PreAmp(ProcessNoise):
    """Class for emulating an analogue pre-amplifier"""
    _settings_noise: SettingsNoise
    __print_device = "switched capacitor"

    def __init__(self, settings_dev: SettingsSC, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def __gen_chop(self, size: int) -> np.ndarray:
        """Generate the chopping clock signal"""
        t = np.arange(0, size, 1) / self._settings.fs_ana
        duty_cycle = self._settings.fs_sc / self._settings.fs_ana
        clk_chop = square(2 * np.pi * t * self._settings.fs_sc, duty=duty_cycle)
        return clk_chop

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
        u_out += self._settings.gain * self._settings.offset
        u_out += self._settings.vcm

        # Adding noise
        if self._settings.noise_en:
            u_out += self._settings.gain * self._gen_noise_real(du.size)

        return self.__voltage_clipping(u_out)
