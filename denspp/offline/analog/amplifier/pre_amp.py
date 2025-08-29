import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, lfilter, square
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, DefaultSettingsNoise


@dataclass
class SettingsAMP:
    """Individual data class to configure the PreAmp
    Attributes:
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
    gain:   float
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


DefaultSettingsAMP = SettingsAMP(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3, gain=40.,
    n_filt=1, f_filt=[0.1, 8e3], f_type="bandpass",
    offset=0e-6,
    f_chop=10e3,
    noise_en=False,
    noise_edev=100e-9
)


class PreAmp(CommonAnalogFunctions):
    _handler_noise: ProcessNoise
    _settings: SettingsAMP

    def __init__(self, settings_dev: SettingsAMP, settings_noise: SettingsNoise=DefaultSettingsNoise):
        """Class for emulating an analogue pre-amplifier
        :param settings_dev:        Dataclass for handling the pre-amplifier
        :param settings_noise:      Dataclass for handling the noise simulation
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._handler_noise = ProcessNoise(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

        # --- Filter properties
        self.__coeffb, self.__coeffa = butter(
            N=self._settings.n_filt,
            Wn=2 * np.array(self._settings.f_filt) / self._settings.fs_ana,
            btype=self._settings.f_type,
            analog=False
        )

    @property
    def get_filter_coeffs(self) -> dict:
        """Getting the filter coefficients"""
        return {'b': self.__coeffb, 'a': self.__coeffa}

    @property
    def vcm(self) -> float:
        return self._settings.vcm

    def __gen_chop(self, size: int) -> np.ndarray:
        """Generate the chopping clock signal"""
        t = np.arange(0, size, 1) / self._settings.fs_ana
        clk_chop = square(2 * np.pi * t * self._settings.f_chop, duty=0.5)
        return clk_chop

    def __noise_generation_circuit(self, size: int) -> np.ndarray:
        """Generating of noise using circuit noise properties"""
        if self._settings.noise_en:
            u_out = self._handler_noise.gen_noise_awgn_dev(size, self._settings.noise_edev)
        else:
            u_out = np.zeros((size,))
        return u_out

    def pre_amp(self, uinp: np.ndarray, uinn: np.ndarray | float) -> np.ndarray:
        """Performs the pre-amplification (single, normal) with input signal
        Args:
            uinp:   Positive input voltage [V]
            uinn:   Negative input voltage [V]
        Returns:
            Corresponding numpy array with output voltage signal
        """
        du = uinp - uinn
        u_out = self._settings.gain * lfilter(b=self.__coeffb, a=self.__coeffa, x=du)
        u_out += self._settings.gain * self._settings.offset
        u_out += self._settings.vcm
        u_out += self._settings.gain * self.__noise_generation_circuit(du.size)
        return self.clamp_voltage(u_out)

    def pre_amp_chopper(self, uinp: np.ndarray, uinn: np.ndarray | float) -> dict:
        """Performs the pre-amplification (single, chopper) with input signal
        Args:
            uinp:   Positive input voltage
            uinn:   Negative input voltage
        Returns:
            Dictionary with results ['out' = Output voltage from pre-amp, 'chop' = chopped voltage signal]
        """
        du = uinp - uinn
        clk_chop = self.__gen_chop(du.size)
        # --- Chopping
        du = (du + self._settings.offset - self._settings.vcm) * clk_chop
        uchp_in = self._settings.vcm + self._settings.gain * du
        uchp_in += self._settings.gain * self.__noise_generation_circuit(du.size)
        # --- Back chopping and Filtering
        u_filt = uchp_in * clk_chop
        u_out = lfilter(self.__coeffb, self.__coeffa, u_filt)
        u_out += self._settings.vcm
        return {'out': self.clamp_voltage(u_out), 'chop': self.clamp_voltage(uchp_in)}
