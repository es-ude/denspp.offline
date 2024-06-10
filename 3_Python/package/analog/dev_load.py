import dataclasses
import numpy as np
from scipy.integrate import cumtrapz
from scipy.constants import Boltzmann, elementary_charge
from package.analog.dev_noise import ProcessNoise, SettingsNoise


@dataclasses.dataclass
class SettingsDEV:
    """Individual data class to configure the electrical device
    Inputs:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor, 'Mem': Memristive (Light)]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
        dev_value:  Value of the selected electrical device
        temp:       Temperature [K]
    """
    type:       str
    fs_ana:     float
    noise_en:   bool
    para_en:    bool
    dev_value:  float
    temp:       float


RecommendedSettingsDEV = SettingsDEV(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    dev_value=10e3,
    temp=300
)

RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
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
    _settings: SettingsDEV
    _dev_type: dict
    __print_device = "electrical behaviour"
    __r_series = 100.0

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self._settings.temp / elementary_charge

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev
        self._dev_type = self._init_dev()

    def _init_dev(self) -> dict:
        """Initialization of simple devices"""
        dev_type = {'R': self.__resistor, 'C': self.__capacitor, 'L': self.__inductor}
        dev_type.update({'Ds': self.__diode_single_barrier, 'Dd': self.__diode_double_barrier})
        dev_type.update({'Mem': self.__memristor_light})
        return dev_type

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
        Returns:
            Corresponding current response
        """
        if isinstance(u_top, float):
            iout = np.zeros((1, ), dtype=float)
        else:
            iout = np.zeros(u_top.shape, dtype=float)

        if self._settings.type in self._dev_type.keys():
            iout = self._dev_type[self._settings.type](u_top, u_bot)
        return iout

    def get_current_density(self, u_top: np.ndarray, u_bot: np.ndarray | float, area: float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
            area:       Area of device [mm^2]
        Returns:
            Corresponding current density response [A/mm^2]
        """
        return self.get_current(u_top, u_bot) / area

    def get_voltage(self, i_in: np.ndarray, u_inn: np.ndarray | float,
                             start_step=1e-3, take_last_value=True) -> np.ndarray:
        """Getting the voltage response from electrical device
        Args:
            i_in:               Applied current input [A]
            u_inn:              Negative input | bottom electrode | reference voltage [V]
            start_step:         Start precision voltage to start iterating the top electrode voltage
            take_last_value:    Option to take the voltage value from last sample (faster)
        Returns:
            Corresponding voltage response
        """
        u_response = np.zeros(i_in.shape)
        idx = 0
        for i0 in i_in:
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
                i1 = self.get_current(u_top, u_bottom)
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
                i1 = self.get_current(u_top, u_bottom)

                # Error Logging
                error.append(_error_mse(i1, i0))
                if len(error) > 1:
                    derror.append(error[-1] - error[-2])
                    error_sign.append(np.sign(derror[-1]) == -1.0)

                # Final Decision (with hyperparameter)
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

    def __resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical resistor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = du / self._settings.dev_value
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_curr(du.size, self._settings.dev_value)
        return i_out

    def __capacitor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical capacitor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = np.zeros(du.shape)
        i_out[1:] = self._settings.dev_value * np.diff(du) * self._settings.fs_ana
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def __inductor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical inductor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = cumtrapz(du, dx=1/self._settings.fs_ana, initial=0) / self._settings.dev_value
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def __diode_single_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
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
        du = u_inp - u_inn
        i_out = is0 * np.exp((du - u_th) / (n0 * self.temperature_voltage))
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def __diode_double_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
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

        du = u_inp - u_inn
        i_out = is0 * (np.exp((du - u_th) / (n0 * self.temperature_voltage)) - 1)
        i_out -= is0 * (np.exp((-du - u_th) / (n0 * self.temperature_voltage)) - 1)
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def __memristor_light(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a memristor (light) with single-side hysterese
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        is0 = 10e-12
        n0 = 9
        u_th = 0.55
        du = u_inp - u_inn

        i_out = is0 * np.exp((du - u_th) / (n0 * self.temperature_voltage))
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out


def __plot_test_results(time: np.ndarray, u_in: np.ndarray, i_in: np.ndarray,
                        mode_current_input: bool, do_ylog=False) -> None:
    """Only for testing"""
    scale_i = 1e3
    scale_u = 1

    signalx = scale_i * i_in if mode_current_input else scale_u * u_in
    signaly = scale_u * u_in if mode_current_input else scale_i * i_in
    label_axisx = 'Voltage U_x [V]' if mode_current_input else 'Current I_x [mA]'
    label_axisy = 'Current I_x [mA]' if mode_current_input else 'Voltage U_x [V]'
    label_legx = 'i_in' if mode_current_input else 'u_in'
    label_legy = 'u_out' if mode_current_input else 'i_out'

    # --- Plotting: Transient signals
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(time[0], time[-1])
    twin1 = axs[0].twinx()
    a = axs[0].plot(time, signalx, 'k', label=label_legx)
    axs[0].set_ylabel(label_axisy)
    axs[0].set_xlabel('Time t [s]')
    b = twin1.plot(time, signaly, 'r', label=label_legy)
    twin1.set_ylabel(label_axisx)
    axs[0].grid()

    # Generate common legend
    lns = a + b
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc=0)

    # --- Plotting: I-U curve
    if mode_current_input:
        if do_ylog:
            axs[1].semilogy(signaly, signalx, 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signaly, signalx, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisx)
        axs[1].set_ylabel(label_axisy)
    else:
        if do_ylog:
            axs[1].semilogy(signalx, signaly, 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signalx, signaly, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisy)
        axs[1].set_ylabel(label_axisx)
    axs[1].grid()

    plt.tight_layout()


def __generate_signal(t_end: float, fs: float, upp: list, fsig: list, uoff=0.0) -> [np.ndarray, np.ndarray]:
    """Generating a signal for testing
    Args:
        t_end:      End of simulation
        fs:         Sampling rate
        upp:        List with amplitude values
        fsig:       List with corresponding frequency
        uoff:       Offset voltage
    """
    t0 = np.linspace(0, t_end, num=int(t_end * fs), endpoint=True)
    uinp = np.zeros(t0.shape) + uoff
    for idx, peak_val in enumerate(upp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * fsig[idx])
    return t0, uinp


# --- TEST CASE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    settings = SettingsDEV(
        type='Mem',
        fs_ana=1000e3,
        noise_en=False,
        para_en=False,
        dev_value=10e3,
        temp=300
    )

    # --- Declaration of input
    do_ylog = True
    t_end = 0.5e-3
    t0, uinp = __generate_signal(0.5e-3, settings.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 2.5)
    uinn = 0.0

    # --- Model declaration
    dev = ElectricalLoad(settings)
    iout = dev.get_current(uinp, uinn)
    iin = 1e-4 * uinp
    uout = dev.get_voltage(iin, uinn, 1e-2)

    # --- Plotting: Current response
    plt.close('all')
    __plot_test_results(t0, uinp-uinn, iout, False, do_ylog)
    __plot_test_results(t0, uout+uinn, iin, True, do_ylog)
    plt.show()
