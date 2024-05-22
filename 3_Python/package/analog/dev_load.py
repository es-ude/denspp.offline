import dataclasses
import numpy as np
from tqdm import tqdm
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


def _error_mbe(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean bias error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(y_pred - y_true) / y_pred.size)
    else:
        error = y_pred - y_true
    return error


def _error_mae(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean absolute error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum(np.abs(y_pred - y_true)) / y_pred.size)
    else:
        error = float(np.abs(y_pred - y_true))
    return error


def _error_mse(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with mean squared error"""
    if isinstance(y_true, np.ndarray):
        error = float(np.sum((y_pred - y_true) ** 2) / y_pred.size)
    else:
        error = float(y_pred - y_true) ** 2
    return error


def _error_tanh(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Calculating the distance-based metric with tanh"""
    if isinstance(y_true, np.ndarray):
        error = float(np.tanh(np.sum(y_pred - y_true) / y_pred.size))
    else:
        error = float(np.tanh(y_pred - y_true))
    return error


class ElectricalLoad(ProcessNoise):
    """Class for emulating an electrical device"""
    _settings_noise: SettingsNoise
    __print_device = "electrical behaviour"
    __r_series = 100.0

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def __dut(self, u_top: np.ndarray | float, u_bottom: float, device_sel: int) -> float:
        """"""
        match device_sel:
            case 0:
                i1 = self.resistor(u_top, u_bottom)
            case 2:
                i1 = self.diode_single_barrier(u_top, u_bottom)
            case 3:
                i1 = self.diode_double_barrier(u_top, u_bottom)
            case _:
                i1 = 0.0
        return i1

    def get_voltage_response(self, i_in: np.ndarray, u_inn: np.ndarray | float, device_sel: int) -> np.ndarray:
        """Getting the voltage response from current input of selected electrical device

        Args:
            i_in:   Applied current input [A]
            u_inn:  Negative input | bottom electrode | reference voltage [V]
            device_sel: Selected electrical device [0: resistor, 1: ...]
        Returns:
            Corresponding voltage response
        """
        u_applied_grob = np.linspace(-2, 2, 21, endpoint=True)

        u_response = np.zeros(i_in.shape)
        idx = 0
        for i0 in tqdm(i_in, ncols=100, desc="Progress: "):
            # --- Step #1: Find minimum (grobe Suche)
            error_step0 = list()
            for u_top in u_applied_grob:
                i1 = self.__dut(u_top, u_inn if isinstance(u_inn, float) else u_inn[idx], device_sel)
                error_step0.append(_error_mse(i1, i0))

            error0 = np.array(error_step0)
            x0 = np.argwhere(error0 == error0.min())
            del error0

            # --- Step #2: Find minimum in range [x0-1, x0+1]
            u_applied_fine = np.linspace(u_applied_grob[x0[0] - 1], u_applied_grob[x0[0] + 1], 1001, endpoint=True)
            error_step1 = list()
            for u_top in u_applied_fine:
                i1 = self.__dut(u_top, u_inn if isinstance(u_inn, float) else u_inn[idx], device_sel)
                error_step1.append(_error_mae(i1, i0))

            error0 = np.array(error_step1)
            x0 = np.argwhere(error0 == error0.min())

            # --- Update
            u_response[idx] = u_applied_fine[x0[0]]
            idx += 1

        return u_response

    def get_voltage_response_v2(self, i_in: np.ndarray, u_inn: np.ndarray | float,
                                device_sel: int, start_step=1e-3, take_last_value=True) -> np.ndarray:
        """Getting the voltage response from current input of selected electrical device

        Args:
            i_in:               Applied current input [A]
            u_inn:              Negative input | bottom electrode | reference voltage [V]
            device_sel:         Selected electrical device [0: resistor, 1: ...]
            start_step:         Start precision voltage to start iterating the top electrode voltage
            take_last_value:    Option to take the voltage value from last sample (faster)
        Returns:
            Corresponding voltage response
        """
        u_response = np.zeros(i_in.shape)
        idx = 0
        for i0 in tqdm(i_in, ncols=100, desc="Progress: "):
            u_bottom = u_inn if isinstance(u_inn, float) else u_inn[idx]
            derror = []
            error = []
            error_sign = []

            # First Step Test (Direction)
            initial_value = u_bottom if idx == 0 and not take_last_value else u_response[idx-1]
            test_value = list()
            test_value.append(initial_value - start_step * (float(np.random.random(1) + 0.5)))
            test_value.append(initial_value - 0.5 * start_step * (float(np.random.random(1) - 0.5)))
            test_value.append(initial_value + 0.5 * start_step * (float(np.random.random(1) - 0.5)))
            test_value.append(initial_value + start_step * (float(np.random.random(1) + 0.5)))

            error0 = list()
            for u_top in test_value:
                i1 = self.__dut(u_top, u_bottom, device_sel)
                error0.append(_error_mse(i1, i0))

            error0 = np.array(error0)
            error0_sign = np.sign(np.diff(error0))
            direction = np.sign(np.sum(error0_sign))
            del error0, error0_sign

            # --- Iteration
            u_top = u_bottom if idx == 0 and not take_last_value else u_response[idx-1]
            step_size = start_step
            step_ite = 0
            do_calc = True
            while do_calc:
                i1 = self.__dut(u_top, u_bottom, device_sel)

                # Error Logging
                error.append(_error_mse(i1, i0))
                if len(error) > 1:
                    derror.append(error[-1] - error[-2])
                    error_sign.append(np.sign(derror[-1]) == -1.0)

                # Final Decision (with hyper-parameter)
                if np.abs(error[-1]) >= 1e-24 and step_ite < 8:
                    u_top -= direction * step_size
                    do_calc = True
                else:
                    do_calc = False

                # Logarithmic Updating Mechanism
                if len(error) > 1:
                    if not error_sign[-1]:
                        u_top += 3 * direction * step_size
                        step_size = 0.1 * step_size
                        step_ite += 1
                        direction = -direction

            # --- Update
            u_response[idx] = u_top
            idx += 1

        return u_response

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
        noise_en=False,
        para_en=False,
        value=10e3
    )

    dev = ElectricalLoad(settings)

    # --- Declaration of input
    t_end = 0.1
    t0 = np.linspace(0, 0.0001, num=int(t_end * settings.fs_ana))

    u_off = 0.0
    u_pp = 1.0
    uinp = u_off + u_pp * np.sin(2 * np.pi * t0 * 10e3)
    uinn = 0.0

    # --- Model declaration
    iout = dev.diode_double_barrier(uinp, uinn)

    iin = 1e-4 * uinp
    uout = dev.get_voltage_response_v2(iin, uinn, 3, 1e-2)

    # --- Plotting: Current response
    plt.close('all')
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(t0[0], t0[-1])
    twin1 = axs[0].twinx()
    axs[0].plot(t0, uinp, 'k', label='u_in')
    axs[0].set_ylabel('Voltage U_x [V]')
    axs[0].set_xlabel('Time t [s]')
    twin1.plot(t0, 1e3 * iout, 'r', label='i_out')
    twin1.set_ylabel('Current I_x [mA]')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(uinp - uinn, 1e3 * iout, 'k')
    axs[1].grid()
    axs[1].set_xlabel('Voltage U_x [V]')
    axs[1].set_ylabel('Current I_x [mA]')
    plt.tight_layout()
    del axs

    # --- Plotting: Voltage response
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(t0[0], t0[-1])
    twin1 = axs[0].twinx()
    axs[0].plot(t0, 1e3 * iin, 'k', label='i_in')
    axs[0].set_ylabel('Current I_x [mA]')
    axs[0].set_xlabel('Time t [s]')
    twin1.plot(t0, uout, 'r', label='i_out')
    twin1.set_ylabel('Voltage U_x [V]')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(uout, 1e3 * iin, 'k')
    axs[1].grid()
    axs[1].set_xlabel('Voltage U_x [V]')
    axs[1].set_ylabel('Current I_x [mA]')
    plt.tight_layout()

    # --- Do Plot
    plt.show()
