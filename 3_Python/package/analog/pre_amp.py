import dataclasses
import numpy as np
from scipy.signal import butter, lfilter, square
from package.data_process.process_noise import noise_real


@dataclasses.dataclass
class SettingsAMP:
    """"Individual data class to configure the PreAmp
    inputs:
    vdd     - Positive supply voltage [V]
    vss     - Negative supply voltage [V]
    fs_ana  - Sampling frequency of input [Hz]
    gain    - Amplification [V/V]
    n_filt  - Order of filter stage []
    f_filt  - Frequency range of filtering [Hz]
    offset  - Offset voltage of the amplifier [V]
    noise   - Enable noise on output [True/False]
    f_chop  - Chopping frequency [Hz] (for chopper)
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
    noise:  bool
    # Chopper properties
    f_chop: float

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


@dataclasses.dataclass
class SettingsNoise:
    """Settings for configuring the pre-amp parasitics
    inputs:
    wgndB  - effective spectral input noise power [dBW/sqrt(Hz)]
    Fc     - Corner frequency of the flicker (1/f) noise [Hz]
    slope  - Alpha coefficient of the flicker noise []
    """
    wgndB: float
    Fc: float
    slope: float


RecommendedSettingsAMP = SettingsAMP(
    vdd=0.6, vss=0.6,
    fs_ana=50e3, gain=40,
    n_filt=1, f_filt=[0.1, 8e3], f_type="bandpass",
    offset=0e-6, noise=False,
    f_chop=10e3
)


RecommendedSettingsNoise = SettingsNoise(
    wgndB=-100,
    Fc=10,
    slope=0.6
)


class PreAmp:
    """Class for emulating an analogue pre-amplifier"""
    def __init__(self, setting: SettingsAMP):
        # --- Settings
        self.settings = setting
        self.settings_noise = RecommendedSettingsNoise
        self.noise_eff_out = 0.0
        self.noise_pp = 0.0
        self.do_output = False
        # --- Filter properties
        self.f_filt = np.array(self.settings.f_filt)
        iir_spk_result = butter(self.settings.n_filt, 2 * self.f_filt / self.settings.fs_ana, self.settings.f_type)
        (self.__b_iir_spk, self.__a_iir_spk) = iir_spk_result[0], iir_spk_result[1]

    def __gen_noise(self, size: int, output: bool) -> np.ndarray:
        """Generate the transient input noise of the amplifier"""
        unoise, self.noise_eff_out, self.noise_pp = noise_real(
            tsize=size,
            fs=self.settings.fs_ana,
            wgndBW=self.settings_noise.wgndB,
            Fc=self.settings_noise.Fc,
            alpha=self.settings_noise.slope
        )
        if output:
            print(f"... effective input noise of pre-amplifier: {1e6 * self.noise_eff_out:.4f} µV")
            print(f"... effective peak-to-peak noise of pre-amplifier: {1e6 * self.noise_pp:.4f} µV")
        return unoise

    def __gen_chop(self, size: int) -> np.ndarray:
        """Generate the chopping clock signal"""
        t = np.arange(0, size, 1) / self.settings.fs_ana
        clk_chop = square(2 * np.pi * t * self.settings.f_chop, duty=0.5)
        return clk_chop

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self.settings.vdd] = self.settings.vdd
        uin[uin < self.settings.vss] = self.settings.vss
        return uin

    def pre_amp(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs the pre-amplification (single, normal) with input signal

        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]

        Returns:
            Test signal
        """
        du = uinp - uinn
        u_out = self.settings.gain * lfilter(b=self.__b_iir_spk, a=self.__a_iir_spk, x=du)
        u_out += self.settings.gain * self.settings.offset
        u_out += self.settings.vcm

        # Adding noise
        if self.settings.noise:
            u_out += self.settings.gain * self.__gen_noise(du.size, self.do_output)
        return self.__voltage_clipping(u_out)

    def pre_amp_chopper(self, uinp: np.ndarray, uinn: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performs the pre-amplification (single, chopper) with input signal

        Args:
            uinp    - Positive input voltage
            uinn    - Negative input voltage
        """
        du = uinp - uinn
        clk_chop = self.__gen_chop(du.size)
        # --- Chopping
        du = (du + self.settings.offset - self.settings.vcm) * clk_chop
        uchp_in = self.settings.vcm + self.settings.gain * du
        # --- Adding noise
        if self.settings.noise:
            uchp_in += self.settings.gain * self.__gen_noise(du.size, self.do_output)
        # --- Back chopping and Filtering
        u_filt = uchp_in * clk_chop
        u_out = lfilter(self.__b_iir_spk, self.__a_iir_spk, u_filt)
        u_out += self.settings.vcm
        return self.__voltage_clipping(u_out), self.__voltage_clipping(uchp_in)
