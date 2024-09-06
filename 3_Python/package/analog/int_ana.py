import dataclasses
import numpy as np
from package.analog.dev_noise import ProcessNoise, SettingsNoise


@dataclasses.dataclass
class SettingsINT:
    """Individual data class to configure an analog voltage integrator
    Params:
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

RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
)


class IntegratorStage(ProcessNoise):
    """Class for emulating an analogue integrator for voltage and current transient signals"""
    _settings: SettingsINT
    _sampling_rate: float
    __print_device = "voltage integrator"

    def __init__(self, settings_dev: SettingsINT, fs: float, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, fs)
        self._sampling_rate = fs
        self._settings = settings_dev

    @property
    def vcm(self) -> float:
        return (self._settings.vdd + self._settings.vss) / 2

    @property
    def tau_active_scale(self) -> float:
        """Getting the time constant tau of integrator"""
        return 1 / self._settings.tau / self._sampling_rate

    def __voltage_clipping(self, uin: np.ndarray | float) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uout = np.zeros(uin.shape) + uin
        if uin.size == 1:
            uout = uout if not uin > self._settings.vdd else self._settings.vdd
            uout = uout if not uin < self._settings.vss else self._settings.vss
        else:
            xpos = np.argwhere(uin > self._settings.vdd)
            xneg = np.argwhere(uin < self._settings.vss)
            uout[xpos] = self._settings.vdd
            uout[xneg] = self._settings.vss
        return uout

    def __noise_generation_resistance(self, size: int) -> np.ndarray:
        """Generating of noise using input resistance"""
        if self._settings.noise_en:
            u_out = self._gen_noise_awgn_volt(size, self._settings.res_in)
        else:
            u_out = np.zeros((size,))
        return u_out

    def __noise_generation_circuit(self, size: int) -> np.ndarray:
        """Generating of noise using circuit noise properties"""
        if self._settings.noise_en:
            u_out = self._gen_noise_awgn_dev(size, self._settings.noise_edev)
        else:
            u_out = np.zeros((size, ))
        return u_out

    def __do_inversion(self, u_int: np.ndarray) -> np.ndarray:
        """Doing the inversion of signal input"""
        u_out = u_int if not self._settings.do_invert else -u_int
        return u_out

    def __do_accumulation_sample(self, x_inp: np.ndarray, x_inn: np.ndarray, scale=1.0) -> np.ndarray:
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

    def __do_accumulation_passive(self, x_inp: np.ndarray, x_inn: np.ndarray, scale=1.0, do_push=False) -> np.ndarray:
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
            u_bot = x_inn[idx] if x_inn.size > 1 else x_inn
            du = scale * self.__do_inversion(u_top - u_bot)
            u_out[idx] = self.__voltage_clipping(u_out[idx-1] + du)
        return u_out

    def __do_accumulation_active(self, x_inp: np.ndarray, x_inn: np.ndarray | float, scale=1.0) -> np.ndarray:
        """Performs an active-accumulation of input signals
        Args:
            x_inp:   Positive input signal
            x_inn:   Negative input signal
            scale:  Scaling value for integration [V/V]
        Returns:
            Numpy array with signal of accumulated input
        """
        u_out = np.zeros(x_inp.shape) + self.vcm
        for idx, u_top in enumerate(x_inp[1:], start=1):
            u_bot = x_inn[idx] if isinstance(x_inn, np.ndarray) else x_inn
            u_int = u_out[idx-1] + scale * self.__do_inversion(u_top - u_bot)
            u_out[idx] = self.__voltage_clipping(u_int)
        return u_out

    def __do_accumulation_resistance(self, u_inp: np.ndarray, u_inn: np.ndarray, scale=1.0) -> np.ndarray:
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
        return self.__voltage_clipping(u_out)

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
        return self.__voltage_clipping(u_out)

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
        return self.__voltage_clipping(u_out)

    def do_opa_curr_integration(self, iin: np.ndarray, uref: np.ndarray) -> np.ndarray:
        """Performs a capacitive passiv-integration behaviour
        Args:
            iin:    Input current [V]
            uref:   Reference voltage of integrator [V]
        Returns:
            Numpy array with voltage signal
        """
        u_out = self.__do_accumulation_active(iin, np.array(0)) + uref
        u_out += self.__noise_generation_circuit(u_out.size)
        return self.__voltage_clipping(u_out)

    def do_cap_curr_integration(self, iin: np.ndarray) -> np.ndarray:
        """Performs a capacitive passiv-integration behaviour
        Args:
            iin:    Input current [V]
        Returns:
            Numpy array with voltage signal
        """
        u_out = self.__do_accumulation_passive(iin, np.array(0))
        return self.__voltage_clipping(u_out)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # --- Definition of Inputs
    settings = RecommendedSettingsINT
    f_smp = 10e3
    t_end = 1
    u_off = 0e-3
    upp = [0.1, 0.02]
    f0 = [3, 10]

    # --- Generation of signals
    time = np.linspace(0, t_end, int(t_end * f_smp), endpoint=True)
    u_inp0 = np.zeros(time.shape) + u_off + settings.vcm
    for idx, peak_value in enumerate(upp):
        u_inp0 += peak_value * np.sin(2 * np.pi * time * f0[idx])
    u_inn0 = np.array(settings.vcm)
    i_in0 = u_inp0 - settings.vcm

    # --- DUT (Test condition)
    dev_test = IntegratorStage(settings, f_smp)
    u_out1 = dev_test.do_ideal_integration(u_inp0, u_inn0)
    u_out2 = dev_test.do_opa_volt_integration(u_inp0, u_inn0)
    u_out3 = dev_test.do_cap_curr_integration(i_in0 / 1000)

    # --- Plotting results
    plt.close('all')
    plt.figure()
    plt.plot(time, u_inp0, 'k', label="input")
    plt.plot(time, u_out1, 'b', label="integration (ideal)")
    plt.plot(time, u_out2, 'g', label="integration (real, OPA)")
    plt.plot(time, u_out3, 'r', label="integration (real, cap)")

    plt.grid()
    plt.legend()
    plt.xlabel("Time t / s")
    plt.ylabel("Voltage U_x / V")

    plt.tight_layout()
    plt.show()
