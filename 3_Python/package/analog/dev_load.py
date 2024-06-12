import dataclasses
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import least_squares


@dataclasses.dataclass
class SettingsDEV:
    """Individual data class to configure the electrical device
    Inputs:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor, 'RDs': Resistive diode]
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


class ElectricalLoad_Handler:
    """Class for emulating an electrical device"""
    _settings: SettingsDEV
    _poly_fit: np.ndarray
    _type_device: dict
    _type_string: dict
    _type_func2reg: dict

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self._settings.temp / elementary_charge

    def __init__(self) -> None:
        self._poly_fit = np.zeros((1, ), dtype=float)

    def _extract_iv_curve_from_regression(self, params_dev: list,
                                          bounds_voltage: list, bounds_current: list,
                                          mode=1, num_points_regression=201) -> [np.ndarray, np.ndarray]:
        """Function for getting the I-V curve from regression
        Args:
            params_dev:             List with parameters from device
            bounds_voltage:         List for voltage limitation / range
            bounds_current:         List for current limitation / range
            num_points_regression:  Number of samples for regression
            mode:                   Mode for approximation (0 = Full, 1 = Symmetric (Neg.), 2 = Symmetric (Pos.
        Returns:
            Two numpy arrays with current and voltage from device
        """
        match mode:
            case 1:
                # --- Get Data
                i_pathn = -np.logspace(bounds_current[1], bounds_current[0], num_points_regression, endpoint=True)
                u_pathn = self._type_func2reg[self._settings.type](-i_pathn, params_dev, np.zeros(i_pathn.shape))
                # --- Concatenate arrays
                i_path = np.concatenate((i_pathn, -np.flipud(i_pathn)[1:]), axis=0)
                u_path = np.concatenate((u_pathn, -np.flipud(u_pathn)[1:]), axis=0)
            case 2:
                # --- Get Data
                i_pathp = np.logspace(bounds_current[0], bounds_current[1], num_points_regression, endpoint=True)
                u_pathp = self._type_func2reg[self._settings.type](i_pathp, params_dev, np.zeros(i_pathp.shape))
                # --- Concatenate arrays
                i_path = np.concatenate((-np.flipud(i_pathp)[1:], i_pathp), axis=0)
                u_path = np.concatenate((-np.flipud(u_pathp)[1:], i_pathp), axis=0)
            case _:
                # --- Get Data
                i_pathp = np.logspace(bounds_current[0], bounds_current[1], num_points_regression, endpoint=True)
                i_path = np.concatenate((-np.flipud(i_pathp[1:]), i_pathp), axis=0)
                u_path = -self._type_func2reg[self._settings.type](i_path, params_dev, np.zeros(i_path.shape))

        # --- Limiting with voltage boundries
        x_start = int(np.argwhere(u_path >= bounds_voltage[0])[0])
        x_stop = int(np.argwhere(u_path >= bounds_voltage[1])[0])
        return i_path[x_start:x_stop], u_path[x_start:x_stop]

    def _do_regression(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float, params: list, bounds_current: list) -> np.ndarray:
        """Performing the behaviour of the device with regression
        Args:
            u_inp:          Positive input voltage [V]
            u_inn:          Negative input voltage [V]
            params:         Device parameter
            bounds_current: List with current limitations
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)

        # --- Start Conditions
        bounds = [10 ** bounds_current[0], 10 ** bounds_current[1]]
        y_initial = 1e-6

        # --- Run optimization
        iout = list()
        for idx, u_sample in enumerate(du):
            sign_pos = u_sample >= 0.0
            y_start = y_initial if idx == 0 else abs(iout[-1])
            result = least_squares(self._type_func2reg[self._settings.type], y_start,
                                   jac='3-point', bounds=(bounds[0], bounds[1]),
                                   args=(params, abs(u_sample)))
            iout.append(result.x[0] if sign_pos else -result.x[0])
        return np.array(iout, dtype=float)

    def _get_params_polyfit(self, params_dev: list,
                            bounds_voltage: list, bounds_current: list,
                            do_test=False, num_poly_order=11, num_points_regression=201,
                            plot_title_prefix='',
                            path2save='') -> None:
        """Function to extract the params of electrical device behaviour with polyfit function
        Args:
            params_dev:             List with parameters from device
            bounds_voltage:         List for voltage limitation / range
            bounds_current:         List for current limitation / range
            do_test:                Performing a test
            num_poly_order:         Order for polynominal fit
            num_points_regression:  Number of samples for fitting regression
            plot_title_prefix:      String for plot title as prefix
            path2save:              String with path to save the figure
        Returns:
            None
        """
        i_path, u_path = self._extract_iv_curve_from_regression(
            params_dev=params_dev,
            bounds_voltage=bounds_voltage,
            bounds_current=bounds_current,
            num_points_regression=num_points_regression,
            mode=1
        )
        self._poly_fit = np.polyfit(x=u_path, y=i_path, deg=num_poly_order)

        # --- Calculating the error-related metric (MSE)
        if do_test:
            u_poly = np.linspace(bounds_voltage[0], bounds_voltage[1], num_points_regression, endpoint=True)
            i_poly = self._type_device[self._settings.type](u_poly, 0.0)
            i_test = self._do_regression(u_poly, 0.0, params_dev, bounds_current)
            self._plot_fit_curve(u_poly, i_poly, i_test, plot_title_prefix)

    def _plot_fit_curve(self, u_poly: np.ndarray, i_poly: np.ndarray, i_reg: np.ndarray,
                        title_prefix='', path2save='') -> None:
        """Plotting the output of the polynominal fit function
        Args:
            u_poly:         Numpy array with voltage from polynom fit (input)
            i_poly:         Numpy array of current response
            i_reg:          Numpy array of current response from regression
            title_prefix:   String with prefix of title
            path2save:      String with path to save the figure
        Returns:
            None
        """
        mse = _error_mse(i_poly, i_reg)

        # --- Plotting
        plt.figure()
        axs = list()
        axs.append(plt.subplot(2, 1, 1))
        axs.append(plt.subplot(2, 1, 2, sharex=axs[0]))
        axs[0].semilogy(u_poly, 1e6 * abs(i_reg), 'k', marker='.', markersize=2, label='Regression')
        axs[0].semilogy(u_poly, 1e6 * abs(i_poly), 'r', marker='.', markersize=2, label='Poly. fit')
        axs[0].grid()
        axs[0].set_ylabel(r'Current $log10(I_F)$ / µA')

        axs[1].plot(u_poly, 1e6 * i_reg, 'k', marker='.', markersize=2, label='Regression')
        axs[1].plot(u_poly, 1e6 * i_poly, 'r', marker='.', markersize=2, label='Poly. fit')
        axs[1].grid()
        axs[1].set_ylabel(r'Current $I_F$ / µA')
        axs[1].legend()
        axs[0].set_title(title_prefix + f"sqrt(MSE) = {1e9 * np.sqrt(mse):.4f} nA")

        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')
        plt.tight_layout()

        #if path2save:
        plt.savefig(join(path2save, "device_iv_charac.svg"), format='svg')

    def print_types(self) -> None:
        """Print electrical types in terminal"""
        print("\n==========================================="
              "\nAvailable types of electrical devices")
        for idx, type in enumerate(self._type_device.keys()):
            print(f"\t#{idx:03d}: {type} = {self._type_string[type]}")

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
        Returns:
            Corresponding current response
        """
        if isinstance(u_top, float) or isinstance(u_top, int):
            iout = np.zeros((1, ), dtype=float)
        else:
            iout = np.zeros(u_top.shape, dtype=float)

        if self._settings.type in self._type_device.keys():
            iout = self._type_device[self._settings.type](u_top, u_bot)
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


# ================================ FUNCTIONS FOR TESTING ===================================
def _generate_signal(t_end: float, fs: float, upp: list, fsig: list, uoff=0.0) -> [np.ndarray, np.ndarray]:
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


def _plot_test_results(time: np.ndarray, u_in: np.ndarray, i_in: np.ndarray,
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
            axs[1].semilogy(signalx, abs(signaly), 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signalx, signaly, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisy)
        axs[1].set_ylabel(label_axisx)
    axs[1].grid()

    plt.tight_layout()