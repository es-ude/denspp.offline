import dataclasses
import numpy as np
from scipy.signal import butter, lfilter, square
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsAMP:
    """Individual data class to configure the PreAmp

    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        gain:       Amplification [V/V]
        n_filt:     Order of filter stage []
        f_filt:     Frequency range of filtering [Hz]
        offset:     Offset voltage of the amplifier [V]
        noise_en:   Enable noise on output [True/False]
        f_chop:     Chopping frequency [Hz] (for chopper)
    """
    vdd:    float
    vss:    float
    fs_ana: float
    # Amplifier characteristics
    gain:   int
    n_filt: int
    f_filt: list
    f_type: str
    offset: float
    noise_en:  bool
    # Chopper properties
    f_chop: float

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsAMP = SettingsAMP(
    vdd=0.6, vss=0.6,
    fs_ana=50e3, gain=40,
    n_filt=1, f_filt=[0.1, 8e3], f_type="bandpass",
    offset=0e-6, noise_en=False,
    f_chop=10e3
)


class PreAmp(ProcessNoise):
    """Class for emulating an analogue pre-amplifier"""
    _settings_noise: SettingsNoise
    __print_device = "pre-amplifier"

    def __init__(self, settings_amp: SettingsAMP, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_amp.fs_ana)
        # --- Settings
        self._settings_dev = settings_amp
        self.vcm = self._settings_dev.vcm

        # --- Filter properties
        self.f_filt = np.array(self._settings_dev.f_filt)
        iir_spk_result = butter(self._settings_dev.n_filt, 2 * self.f_filt / self._settings_dev.fs_ana, self._settings_dev.f_type)
        (self.__b_iir_spk, self.__a_iir_spk) = iir_spk_result[0], iir_spk_result[1]

    def __gen_chop(self, size: int) -> np.ndarray:
        """Generate the chopping clock signal"""
        t = np.arange(0, size, 1) / self._settings_dev.fs_ana
        clk_chop = square(2 * np.pi * t * self._settings_dev.f_chop, duty=0.5)
        return clk_chop

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self._settings_dev.vdd] = self._settings_dev.vdd
        uin[uin < self._settings_dev.vss] = self._settings_dev.vss
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
        u_out = self._settings_dev.gain * lfilter(b=self.__b_iir_spk, a=self.__a_iir_spk, x=du)
        u_out += self._settings_dev.gain * self._settings_dev.offset
        u_out += self._settings_dev.vcm

        # Adding noise
        if self._settings_dev.noise_en:
            u_out += self._settings_dev.gain * self._gen_noise_real(du.size)
        return self.__voltage_clipping(u_out)

    def pre_amp_chopper(self, uinp: np.ndarray, uinn: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performs the pre-amplification (single, chopper) with input signal

        Args:
            uinp:   Positive input voltage
            uinn:   Negative input voltage
        """
        du = uinp - uinn
        clk_chop = self.__gen_chop(du.size)
        # --- Chopping
        du = (du + self._settings_dev.offset - self._settings_dev.vcm) * clk_chop
        uchp_in = self._settings_dev.vcm + self._settings_dev.gain * du
        # --- Adding noise
        if self._settings_dev.noise_en:
            uchp_in += self._settings_dev.gain * self._gen_noise_real(du.size)
        # --- Back chopping and Filtering
        u_filt = uchp_in * clk_chop
        u_out = lfilter(self.__b_iir_spk, self.__a_iir_spk, u_filt)
        u_out += self._settings_dev.vcm

        return self.__voltage_clipping(u_out), self.__voltage_clipping(uchp_in)
