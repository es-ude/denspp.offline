import dataclasses
import numpy as np
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise
from scipy.integrate import cumtrapz
from scipy.constants import Boltzmann, elementary_charge


@dataclasses.dataclass
class SettingsDEV:
    """"Individual data class to configure the electrical device

    Args:
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
        value:      Value of the selected electrical device
    """
    fs_ana:     float
    noise_en:   bool
    para_en:    bool
    value:      float


RecommendedSettingsDEV = SettingsDEV(
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    value=10e3
)


class ElectricalLoad(ProcessNoise):
    """Class for emulating an electrical device"""
    _settings_noise: SettingsNoise
    __print_device = "electrical behaviour"
    __r_series = 100.0

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical resistor

        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]

        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn

        i_out = du / self._settings.value
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn(du.size, False)
        return i_out

    def capacitor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical capacitor

        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]

        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn

        i_out = np.zeros(du.shape)
        i_out[1:] = self._settings.value * np.diff(du) * self._settings.fs_ana
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn(du.size, False)
        return i_out

    def inductor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical inductor

        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]

        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = cumtrapz(du, dx=1/self._settings.fs_ana, initial=0) / self._settings.value
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn(du.size, False)
        return i_out

    def diode_single_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a diode with single-side barrier

        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]

        Returns:
            Corresponding current signal
        """
        is0 = 1e-12
        n0 = 1.4
        u_th = 0.25

        u_kb = Boltzmann * 300 / elementary_charge
        du = u_inp - u_inn
        i_out = is0 * np.exp((du - u_th) / (n0 * u_kb))
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn(du.size, False)
        return i_out

    def diode_double_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a diode with double-side barrier

        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]

        Returns:
            Corresponding current signal
        """
        is0 = 1e-12
        n0 = 1.4
        u_th = 0.5

        u_kb = Boltzmann * 300 / elementary_charge
        du = u_inp - u_inn
        i_out = is0 * (np.exp((du - u_th) / (n0 * u_kb)) - 1)
        i_out -= is0 * (np.exp((-du - u_th) / (n0 * u_kb)) - 1)
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn(du.size, False)
        return i_out


# --- TEST CASE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    settings = SettingsDEV(
        fs_ana=50e3,
        noise_en=True,
        para_en=False,
        value=1e-15
    )

    dev = ElectricalLoad(settings)

    # --- Declaration of input
    t_end = 0.1
    t0 = np.linspace(0, 0.0001, num=int(t_end * settings.fs_ana))
    uinp = 100e-3 * np.sin(2 * np.pi * t0 * 10e3)
    uinn = 0.0
    iout = dev.diode_double_barrier(uinp, uinn)

    # --- Plotting
    plt.figure()
    axs = [plt.subplot(2, 1, idx+1) for idx in range(2)]

    axs[0].set_xlim(t0[0], t0[-1])
    twin1 = axs[0].twinx()
    axs[0].plot(t0, uinp, 'k', label='u_in')
    axs[0].set_ylabel('Voltage U_x [V]')
    axs[0].set_xlabel('Time t [s]')
    twin1.plot(t0, 1e3 * iout, 'r', label='i_out')
    twin1.set_ylabel('Current I_x [mA]')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(uinp-uinn, 1e3 * iout, 'k')
    axs[1].grid()
    axs[1].set_xlabel('Voltage U_x [V]')
    axs[1].set_ylabel('Current I_x [mA]')

    plt.tight_layout()
    plt.show()
