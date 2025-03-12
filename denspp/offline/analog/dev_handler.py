from dataclasses import dataclass
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import least_squares, curve_fit

from denspp.offline.structure_builder import init_project_folder
from denspp.offline.plot_helper import save_figure
from denspp.offline.metric.data_numpy import calculate_error_rae, calculate_error_mse


@dataclass
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


def _calc_error(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    return calculate_error_rae(y_pred, y_true)


def _raise_voltage_violation(du: np.ndarray | float, range_volt: list) -> None:
    """Checking differential voltage input for violation of voltage range for given branch"""
    violation_dwn = np.count_nonzero(du < range_volt[0], axis=0)
    violation_up = np.count_nonzero(du > range_volt[1], axis=0)

    if violation_up or violation_dwn:
        val = du.min if violation_dwn else du.max
        limit = range_volt[slice(0)] if violation_dwn else range_volt[slice(1)]
        addon = f'(Upper limit)' if not violation_dwn else '(Downer limit)'
        warn(f"--- Warning: Voltage Range Violation {addon}! With {val} of {limit} ---")


class ElectricalLoadHandler:
    """Class for emulating an electrical device"""
    _settings: SettingsDEV
    _poly_fit: np.ndarray
    _curve_fit: np.ndarray
    _approx_fit: np.ndarray
    _fit_options: list
    _type_device: dict
    _type_string: dict
    _type_func2reg: dict
    _type_func2cur: dict
    # Local limitations
    _bounds_curr: list
    _bounds_volt: list
    _params_used: list

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self._settings.temp / elementary_charge

    def __init__(self) -> None:
        init_project_folder()
        self._init_class()

    def _init_class(self) -> None:
        self._poly_fit = np.zeros((1, ), dtype=float)
        self._curve_fit = np.zeros((1, ), dtype=float)
        self._approx_fit = np.zeros((1, ), dtype=float)
        self._fit_options = [1, 1001]

        self._bounds_curr = [-14, -3]
        self._bounds_volt = [0.0, +6]
        self._params_used = [1e0]

    def _extract_iv_curve_from_regression(self, params_dev: list,
                                          bounds_voltage: list, bounds_current: list,
                                          mode_fit: int=0) -> [np.ndarray, np.ndarray]:
        """Function for getting the I-V curve from regression
        Args:
            params_dev:             List with parameters from device
            bounds_voltage:         List for voltage limitation / range
            bounds_current:         List for current limitation / range
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
        Returns:
            Two numpy arrays with current and voltage from device
        """

        u_path = np.zeros((1,), dtype=float)
        i_path = np.zeros((1,), dtype=float)

        if self._settings.type in self._type_func2reg:
            match mode_fit:
                case 0:
                    # --- Get Data (Given Range)
                    i_path = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    u_path = -self._type_func2reg[self._settings.type](i_path, params_dev, np.zeros(i_path.shape))
                case 1:
                    # --- Get Data (Full Range)
                    i_pathp = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    i_path = np.concatenate((-np.flipud(i_pathp[1:]), i_pathp), axis=0)
                    u_path = -self._type_func2reg[self._settings.type](i_path, params_dev, np.zeros(i_path.shape))
                case 2:
                    # --- Get Data (Symmetric, Pos. mirrored)
                    i_pathn = -np.logspace(bounds_current[1], bounds_current[0], self._fit_options[1], endpoint=True)
                    u_pathn = self._type_func2reg[self._settings.type](-i_pathn, params_dev, np.zeros(i_pathn.shape))
                    # --- Concatenate arrays
                    i_path = np.concatenate((i_pathn, -np.flipud(i_pathn)[1:]), axis=0)
                    u_path = np.concatenate((u_pathn, -np.flipud(u_pathn)[1:]), axis=0)
                case 3:
                    # --- Get Data (Symmetric, Neg. mirrored)
                    i_pathp = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    u_pathp = self._type_func2reg[self._settings.type](i_pathp, params_dev, np.zeros(i_pathp.shape))
                    # --- Concatenate arrays
                    i_path = np.concatenate((-np.flipud(i_pathp)[1:], i_pathp), axis=0)
                    u_path = np.concatenate((-np.flipud(u_pathp)[1:], i_pathp), axis=0)
        else:
            warn("Using normal device equation for getting I-V-behaviour")
            u_path = np.linspace(bounds_voltage[0], bounds_voltage[1], self._fit_options[1], endpoint=True)
            i_path = self.get_current(u_path, 0.0)

        # --- Limiting with voltage boundaries
        x_start = int(np.argwhere(u_path >= bounds_voltage[0])[0])
        x_stop = int(np.argwhere(u_path >= bounds_voltage[1])[0])
        return i_path[x_start:x_stop], u_path[x_start:x_stop]

    def _do_regression(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float,
                       params: list, bounds_current: list) -> np.ndarray:
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
        y_initial = 2 * bounds[0]

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

    def _get_params_curve_fit(self, params_dev: list, mode_fit: int,
                              do_test: bool=False, do_plot: bool=True,
                              plot_title_prefix: str='', path2save: str='') -> float:
        """Function to extract the params of electrical device behaviour with curve fitting
        Args:
            params_dev:             List with parameters from device
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
            do_test:                Performing a test
            do_plot:                Plotting the results of regression and polynom fitting
            plot_title_prefix:      String for plot title as prefix
            path2save:              String with path to save the figure
        Returns:
            Floating value with Relative Squared Error
        """
        i_path, u_path = self._extract_iv_curve_from_regression(
            params_dev=params_dev,
            bounds_voltage=self._bounds_volt,
            bounds_current=self._bounds_curr,
            mode_fit=mode_fit
        )
        self._curve_fit = curve_fit(f=self._type_func2cur[self._settings.type], xdata=u_path, ydata=i_path)[0]

        # --- Calculating the metric
        if do_test:
            u_poly = np.linspace(self._bounds_volt[0], self._bounds_volt[1], self._fit_options[1], endpoint=True)
            i_poly = self._type_device[self._settings.type](u_poly, 0.0)
            i_test = self._do_regression(u_poly, 0.0, params_dev, self._bounds_curr)
            error = _calc_error(i_poly, i_test)
            if do_plot:
                self._plot_fit_curve(
                    u_poly, i_poly, i_test, metric=['1e3 * RAE', 1e3 * error],
                    title_prefix=plot_title_prefix, path2save=path2save
                )
        else:
            error = -1.0
        return error

    def _get_params_polyfit(self, params_dev: list, mode_fit: int=0,
                            do_test: bool=False, do_plot: bool=False,
                            plot_title_prefix: str='', path2save: str='') -> float:
        """Function to extract the params of electrical device behaviour with polyfit function
        Args:
            params_dev:             List with parameters from device
            do_test:                Performing a test
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
            do_plot:                Plotting the results of regression and polynom fitting
            plot_title_prefix:      String for plot title as prefix
            path2save:              String with path to save the figure
        Returns:
            Floating value with Relative Squared Error
        """
        i_path, u_path = self._extract_iv_curve_from_regression(
            params_dev=params_dev,
            bounds_voltage=self._bounds_volt,
            bounds_current=self._bounds_curr,
            mode_fit=mode_fit
        )
        if not i_path.size == 1:
            self._poly_fit = np.polyfit(x=u_path, y=i_path, deg=self._fit_options[0])

            u_poly = np.linspace(self._bounds_volt[0], self._bounds_volt[1], self._fit_options[1], endpoint=True)
            i_poly = self._type_device[self._settings.type](u_poly, 0.0)

            # --- Calculating the metric
            if do_test and self._settings.type != 'R':
                i_test = self._do_regression(u_poly, 0.0, params_dev, self._bounds_curr)
                error = _calc_error(i_poly, i_test)
            else:
                i_test = np.zeros_like(u_poly)
                error = -1.0

            self._plot_fit_curve(
                u_poly, i_poly, i_test, metric=['1e3 * RAE', 1e3 * error],
                title_prefix=plot_title_prefix, path2save=path2save, show_plot=do_plot
            )
        else:
            error = -1.0

        return error

    def _plot_fit_curve(self, u_poly: np.ndarray, i_poly: np.ndarray, i_reg: np.ndarray,
                        metric: list=(), title_prefix: str='', path2save: str='', show_plot: bool=False) -> None:
        """Plotting the output of the polynomial fit function
        Args:
            u_poly:         Numpy array with voltage from polynom fit (input)
            i_poly:         Numpy array of current response
            i_reg:          Numpy array of current response from regression
            title_prefix:   String with prefix of title
            path2save:      String with path to save the figure
            show_plot:      Showing and blocking the plots [Default: False]
        Returns:
            None
        """
        plt.figure()
        plt.tight_layout()

        axs = list()
        axs.append(plt.subplot(2, 1, 1))
        axs.append(plt.subplot(2, 1, 2, sharex=axs[0]))
        axs[0].semilogy(u_poly, 1e6 * abs(i_poly), 'r', marker='.', markersize=2, label='Poly. fit')
        axs[0].grid()
        axs[0].set_ylabel(r'Current $\log_{10}(I_F)$ / µA')

        axs[1].plot(u_poly, 1e6 * i_poly, 'r', marker='.', markersize=2, label='Poly. fit')
        axs[1].grid()
        axs[1].set_ylabel(r'Current $I_F$ / µA')
        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')

        if not np.array_equal(i_reg, np.zeros_like(i_reg)):
            axs[0].semilogy(u_poly, 1e6 * abs(i_reg), 'k', marker='.', markersize=2, label='Regression')
            axs[1].plot(u_poly, 1e6 * i_reg, 'k', marker='.', markersize=2, label='Regression')
            axs[1].legend()

        # --- Add figure title
        if not len(metric) == 0:
            axs[0].set_title(title_prefix + f"{metric[0]} = {metric[1]:.4f} @ N_Poly = {self._fit_options[0]}")

        if path2save:
            save_figure(plt, path2save, 'device_iv_charac.svg')
        if show_plot:
            plt.show(block=True)

    def _find_best_poly_order(self, order_start: int, order_stop: int,
                              bounds_voltage: list, params_dev: list,
                              show_plots: bool=False, mode_fit: int=0) -> None:
        """Finding the best polynomial order for fitting
        Args:
            order_start:    Integer value with starting order number
            order_stop:     Integer value with stoping order number
            bounds_voltage: Voltage limitation for fitting
            params_dev:     List
            show_plots:     Showing plots of each run
            mode_fit:       Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
        Returns:
            None
        """
        print("\n=====================================================")
        print("Searching the best poly order with minimal error")
        print("=====================================================")
        order_search = [idx for idx in range(order_start, order_stop+1)]
        error_search = []
        for idx, order in enumerate(order_search):
            self.change_options_fit(order, self._fit_options[1])
            error = self._get_params_polyfit(
                params_dev=params_dev,
                do_test=True, do_plot=show_plots,
                mode_fit=mode_fit
            )
            error_search.append(error)
            print(f"#{idx:02d}: order = {order:02d} --> Error = {error}")

        # --- Finding best order
        error_search = np.array(error_search)
        xmin = np.argwhere(error_search == error_search.min()).flatten()
        print(f"\nBest solution: Order = {np.array(order_search)[xmin]} with an error of {error_search[xmin]}!")
        print("TEST")

    def plot_fit_curve(self, find_best_order: bool=False, show_plots: bool=True,
                       order_start: int=2, order_stop: int=18, mode_fit: int=0) -> None:
        """Plotting the output of the polynom fit function
        Args:
            find_best_order:    Find the best poly.-fit order
            show_plots:         Showing plots of each run
            order_start:        Integer value for starting search (best polynom order)
            order_stop:         Integer value for stopping search (best polynom order)
            mode_fit:           Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
        Returns:
            None
        """
        if len(self._params_used) == 1 and self._params_used[0] == 1.0:
            self.get_current(0.0, 0.0)

        if not find_best_order:
            self._get_params_polyfit(
                params_dev=self._params_used,
                do_test=True, do_plot=show_plots,
                mode_fit=mode_fit
            )
        else:
            self._find_best_poly_order(
                order_start=order_start, order_stop=order_stop,
                params_dev=self._params_used,
                bounds_voltage=self._bounds_volt,
                show_plots=show_plots, mode_fit=mode_fit
            )

    def change_boundary_current(self, downer_limit: float, upper_limit: float) -> None:
        """Redefining the current limits for polynom fitting of I-V behaviour of electrical devices
        Args:
            upper_limit:    Exponential integer for upper current limit
            downer_limit:   Exponential integer for downer current limit
        """
        self._bounds_curr = [downer_limit, upper_limit]

    def change_boundary_voltage(self, downer_limit: float, upper_limit: float) -> None:
        """Redefining the voltage limits for polynom fitting of I-V behaviour of electrical devices
        Args:
            upper_limit:    Exponential integer for upper voltage limit
            downer_limit:   Exponential integer for downer voltage limit
        """
        self._bounds_volt = [downer_limit, upper_limit]

    def change_options_fit(self, poly_order: int, num_points_fit: int) -> None:
        """Redefining the options for polynom fitting of I-V behaviour of electrical devices
        Args:
            poly_order:     Order of the polynom fit
            num_points_fit: Exponential integer for downer voltage limit
        """
        self._fit_options = [poly_order, num_points_fit]

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

        method = [method for method in self._type_device.keys() if method == self._settings.type]
        if len(method):
            iout = self._type_device[self._settings.type](u_top, u_bot)
        else:
            raise Exception("Error: Model not available - Please check!")
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
                    start_value: float=0.0, start_step: float=1e-3, take_last_value: bool=True) -> np.ndarray:
        """Getting the voltage response from electrical device
        Args:
            i_in:               Applied current input [A]
            u_inn:              Negative input | bottom electrode | reference voltage [V]
            start_value:        Starting value [V]
            start_step:         Start precision voltage to start iterating the top electrode voltage
            take_last_value:    Option to take the voltage value from last sample (faster)
        Returns:
            Corresponding voltage response
        """
        if self._settings.type == 'R':
            return i_in * self._params_used[0]
        else:
            u_response = np.zeros(i_in.shape) + start_value
            idx = 0
            for i0 in i_in:
                u_bottom = u_inn if isinstance(u_inn, float) else u_inn[idx]
                derror = []
                error = []
                error_sign = []

                # First Step Test (Direction)
                initial_value = start_value if idx == 0 and not take_last_value else u_response[idx-1]
                test_value = list()
                test_value.append(initial_value - start_step * (float(np.random.random(1) + 0.5)))
                test_value.append(initial_value - 0.5 * start_step * (float(np.random.random(1) - 0.5)))
                test_value.append(initial_value + 0.5 * start_step * (float(np.random.random(1) - 0.5)))
                test_value.append(initial_value + start_step * (float(np.random.random(1) + 0.5)))

                error0 = list()
                for u_top in test_value:
                    i1 = self.get_current(u_top, u_bottom)
                    error0.append(calculate_error_mse(i1, i0))

                error0 = np.array(error0)
                error0_sign = np.sign(np.diff(error0))
                direction = np.sign(np.sum(error0_sign))
                del error0, error0_sign

                # --- Iteration
                u_top = start_value if idx == 0 and not take_last_value else u_response[idx-1]
                step_size = start_step
                step_ite = 0
                do_calc = True
                while do_calc:
                    i1 = self.get_current(u_top, u_bottom)

                    # Error Logging
                    error.append(calculate_error_mse(i1, i0))
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
