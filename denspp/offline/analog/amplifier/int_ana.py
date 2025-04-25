import numpy as np
from dataclasses import dataclass
from denspp.offline.analog.common_func import CommonAnalogFunctions
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclass
class SettingsINT:
    """Individual data class to configure an analog voltage integrator
    Attributes:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        vmargin:    Margin range from supply voltage for non-idealities [V]
        fs_ana:     Sampling frequency of input [Hz]
        tau:        Time constant of integrator circuit [s]
        res_in:     Input resistance of the circuit [Ohm]
        offset_v:   Offset voltage of the amplifier [V]
        offset_i:   Offset current of the amplifier [V]
        do_invert:  Do inversion of integration output [True / False]
        noise_en:   Enable noise on output [True / False]
        noise_edev: Spectal noise voltage density of circuit [V/sqrt(Hz)]
    """
    vdd:        float
    vss:        float
    # Amplifier characteristics
    tau:        float
    res_in:     float
    offset_v:   float
    offset_i:   float
    do_invert:  bool
    # Modes for activating non-idealities
    noise_en:   bool
    noise_edev:  float

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2

    @property
    def u_error(self) -> float:
        return -(self.offset_v + self.offset_i * self.res_in)

    @property
    def u_supply_range(self) -> float:
        return self.vdd - self.vss


RecommendedSettingsINT = SettingsINT(
    vdd=0.6, vss=-0.6,
    tau=100e-3,
    res_in=10e3,
    offset_v=1e-3,
    offset_i=1e-9,
    do_invert=False,
    noise_en=True,
    noise_edev=10e-9
)


class IntegratorStage(CommonAnalogFunctions):
    _handler_noise: ProcessNoise
    _settings: SettingsINT
    _sampling_rate: float
    __print_device = "voltage integrator"

    def __init__(self, settings_dev: SettingsINT, fs: float, settings_noise: SettingsNoise=RecommendedSettingsNoise):
        """Class for emulating an analogue integrator for voltage and current transient signals
        :param settings_dev:        Dataclass for handling the delay amplifier
        :param fs:                  Sampling frequency [Hz]
        :param settings_noise:      Dataclass for handling the noise simulation
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._handler_noise = ProcessNoise(settings_noise, fs)
        self._sampling_rate = fs
        self._settings = settings_dev

    @property
    def tau_active_scale(self) -> float:
        """Getting the time constant tau of integrator"""
        return 1 / self._settings.tau / self._sampling_rate

    def __noise_generation_resistance(self, size: int) -> np.ndarray:
        """Generating of noise using input resistance"""
        if self._settings.noise_en:
            u_out = self._handler_noise.gen_noise_awgn_volt(size, self._settings.res_in)
        else:
            u_out = np.zeros((size,))
        return u_out

    def __noise_generation_circuit(self, size: int) -> np.ndarray:
        """Generating of noise using circuit noise properties"""
        if self._settings.noise_en:
            u_out = self._handler_noise.gen_noise_awgn_dev(size, self._settings.noise_edev)
        else:
            u_out = np.zeros((size, ))
        return u_out

    def __do_inversion(self, u_int: np.ndarray) -> np.ndarray:
        """Doing the inversion of signal input"""
        u_out = u_int if not self._settings.do_invert else -u_int
        return u_out

    @staticmethod
    def __do_accumulation_sample(x_inp: np.ndarray, x_inn: np.ndarray, scale: float=1.0) -> np.ndarray:
        """Performs an accumulation of input signals
        Args:
            x_inp:      Positive input signal
            x_inn:      Negative input signal
            scale:      Scaling value for integration [V/V]
        Returns:
            Numpy array with accumulated input
        """
        u_out = np.sum(x_inp - x_inn, axis=0) * scale
        return u_out

    def __do_accumulation_passive(self, x_inp: np.ndarray, x_inn: np.ndarray, scale: float=1.0, do_push: bool=False) -> np.ndarray:
        """Performs a passive-accumulation of input signals
        Args:
            x_inp:      Positive input signal
            x_inn:      Negative input signal
            scale:      Scaling value for integration [V/V]
            do_push:    Element is push- or pull-element
        Returns:
            Numpy array with signal of accumulated input
        """
        u_out = np.zeros(x_inp.shape) + (self._settings.vss if not do_push else self._settings.vdd)
        for idx, u_top in enumerate(x_inp[1:], start=1):
            u_bot = (x_inn[idx] if x_inn.size > 1 else x_inn) if isinstance(x_inn, np.ndarray) else x_inn
            du = scale * self.__do_inversion(u_top - u_bot)
            u_out[idx] = self.clamp_voltage(u_out[idx - 1] + du)
        return u_out

    def __do_accumulation_active(self, x_inp: np.ndarray, x_inn: np.ndarray | float, scale: float=1.0) -> np.ndarray:
        """Performs an active-accumulation of input signals
        Args:
            x_inp:   Positive input signal
            x_inn:   Negative input signal
            scale:  Scaling value for integration [V/V]
        Returns:
            Numpy array with signal of accumulated input
        """
        u_out = np.zeros(x_inp.shape) + self._settings.vcm
        for idx, u_top in enumerate(x_inp[1:], start=1):
            u_bot = (x_inn[idx] if x_inn.size > 1 else x_inn) if isinstance(x_inn, np.ndarray) else x_inn
            u_int = u_out[idx-1] + scale * self.__do_inversion(u_top - u_bot)
            u_out[idx] = self.clamp_voltage(u_int)
        return u_out

    def __do_accumulation_resistance(self, u_inp: np.ndarray, u_inn: np.ndarray, scale: float=1.0) -> np.ndarray:
        """Performs an active-accumulation of input signals with additional input resistance
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
            scale:  Scaling value for integration [V/V]
        Returns:
            Numpy array with voltage signal of accumulated input
        """
        u_top = u_inp + self.__noise_generation_resistance(u_inp.size)
        return self.__do_accumulation_active(u_top, u_inn, scale)

    def do_ideal_integration_sample(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performs an ideal active-integration behaviour
        Args:
            u_inp:   Positive input voltage or current [V | A]
            u_inn:   Negative input voltage [V]
        Returns:
            Numpy array with voltage sample
        """
        u_n = self.__noise_generation_circuit(u_inp.size)
        u_out = self.__do_accumulation_sample(u_inp + u_n, u_inn, self.tau_active_scale)
        return self.clamp_voltage(u_out)

    def do_ideal_integration(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performs an ideal active-integration behaviour
        Args:
            u_inp:   Positive input voltage or current [V | A]
            u_inn:   Negative input voltage [V]
        Returns:
            Numpy array with voltage signal
        """
        u_out = self.__do_accumulation_active(u_inp, u_inn, self.tau_active_scale)
        u_out += self.__noise_generation_circuit(u_out.size)
        return self.clamp_voltage(u_out)

    def do_opa_volt_integration(self, u_inp: np.ndarray, u_inn: np.ndarray) -> np.ndarray:
        """Performs an active-integration behaviour using operational amplifiers (OPA)
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Numpy array with voltage signal
        """
        u_top = u_inp + self._settings.u_error
        u_out = self.__do_accumulation_resistance(u_top, u_inn, self.tau_active_scale) + self._settings.offset_v
        u_out += self.__noise_generation_circuit(u_out.size)
        return self.clamp_voltage(u_out)

    def do_opa_curr_integration(self, iin: np.ndarray, uref: np.ndarray) -> np.ndarray:
        """Performs a capacitive passiv-integration behaviour
        Args:
            iin:    Input current [V]
            uref:   Reference voltage of integrator [V]
        Returns:
            Numpy array with voltage signal
        """
        u_out = self.__do_accumulation_active(iin, 0.0) + uref
        u_out += self.__noise_generation_circuit(u_out.size)
        return self.clamp_voltage(u_out)

    def do_cap_curr_integration(self, iin: np.ndarray) -> np.ndarray:
        """Performs a capacitive passiv-integration behaviour
        Args:
            iin:    Input current [V]
        Returns:
            Numpy array with voltage signal
        """
        u_out = self.__do_accumulation_passive(iin, 0.0)
        return self.clamp_voltage(u_out)
