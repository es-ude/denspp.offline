import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, lfilter, square
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclass
class SettingsAMP:
    """Individual data class to configure the PreAmp

    Params:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        gain:       Amplification [V/V]
        n_filt:     Order of filter stage []
        f_filt:     Frequency range of filtering [Hz]
        offset:     Offset voltage of the amplifier [V]
        f_chop:     Chopping frequency [Hz] (for chopper)
        noise_en:   Enable noise on output [True/False]
        noise_edev: Input voltage noise spectral density [V/sqrt(Hz)]
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
    # Chopper properties
    f_chop: float
    # Noise properties
    noise_en: bool
    noise_edev: float

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsAMP = SettingsAMP(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3, gain=40,
    n_filt=1, f_filt=[0.1, 8e3], f_type="bandpass",
    offset=0e-6,
    f_chop=10e3,
    noise_en=False,
    noise_edev=100e-9
)


class PreAmp(CommonAnalogFunctions):
    handler_noise: ProcessNoise
    settings: SettingsAMP
    __print_device = "pre-amplifier"

    def __init__(self, settings_amp: SettingsAMP, settings_noise: SettingsNoise=RecommendedSettingsNoise):
        """Class for emulating an analogue pre-amplifier"""
        super().__init__(settings_amp)
        self.handler_noise = ProcessNoise(settings_noise, settings_amp.fs_ana)
        self.settings = settings_amp

        # --- Filter properties
        f_filt = np.array(self.settings.f_filt)
        iir_spk_result = butter(self.settings.n_filt, 2 * f_filt / self.settings.fs_ana,
                                self.settings.f_type)
        (self.__b_iir_spk, self.__a_iir_spk) = iir_spk_result[0], iir_spk_result[1]

    @property
    def vcm(self) -> float:
        return self._settings.vcm

    def __gen_chop(self, size: int) -> np.ndarray:
        """Generate the chopping clock signal"""
        t = np.arange(0, size, 1) / self.settings.fs_ana
        clk_chop = square(2 * np.pi * t * self.settings.f_chop, duty=0.5)
        return clk_chop

    def __noise_generation_circuit(self, size: int) -> np.ndarray:
        """Generating of noise using circuit noise properties"""
        if self.settings.noise_en:
            u_out = self.handler_noise.gen_noise_awgn_dev(size, self.settings.noise_edev)
        else:
            u_out = np.zeros((size,))
        return u_out

    def pre_amp(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs the pre-amplification (single, normal) with input signal
        Args:
            uinp:   Positive input voltage [V]
            uinn:   Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage signal
        """
        du = uinp - uinn
        u_out = self.settings.gain * lfilter(b=self.__b_iir_spk, a=self.__a_iir_spk, x=du)
        u_out += self.settings.gain * self.settings.offset
        u_out += self.settings.vcm
        u_out += self.settings.gain * self.__noise_generation_circuit(du.size)
        return self.voltage_clipping(u_out)

    def pre_amp_chopper(self, uinp: np.ndarray, uinn: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performs the pre-amplification (single, chopper) with input signal
        Args:
            uinp:   Positive input voltage
            uinn:   Negative input voltage
        Returns:
            Tuple with two numpy arrays [u_out = Output voltage from pre-amp, u_chp = chopped voltage signal]
        """
        du = uinp - uinn
        clk_chop = self.__gen_chop(du.size)
        # --- Chopping
        du = (du + self.settings.offset - self.settings.vcm) * clk_chop
        uchp_in = self.settings.vcm + self.settings.gain * du
        uchp_in += self.settings.gain * self.__noise_generation_circuit(du.size)
        # --- Back chopping and Filtering
        u_filt = uchp_in * clk_chop
        u_out = lfilter(self.__b_iir_spk, self.__a_iir_spk, u_filt)
        u_out += self.settings.vcm

        return self.voltage_clipping(u_out), self.voltage_clipping(uchp_in)
