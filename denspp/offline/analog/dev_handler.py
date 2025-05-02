import numpy as np
from dataclasses import dataclass
from logging import getLogger
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import least_squares, curve_fit

from denspp.offline.plot_helper import scale_auto_value, save_figure
from denspp.offline.metric.data_numpy import calculate_error_rae, calculate_error_mse


@dataclass
class SettingsDEV:
    """Individual data class to configure the electrical device
    Attributes:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor, 'RDs': Resistive diode]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
        use_mode:   Mode for getting the electrical signals [0: regression, 1: equation, 2: polynom fitting]
        dev_value:  Dictionary with device parameters
        temp:       Temperature [K]
    """
    type:       str
    fs_ana:     float
    noise_en:   bool
    para_en:    bool
    use_mode:   int
    dev_value:  dict
    temp:       float

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self.temp / elementary_charge


class ElectricalLoadHandler:
    _settings: SettingsDEV
    _type_device: dict
    # --- Fitting options
    _fit_done: bool
    _fit_options: list
    # --- Electrical Signal limitations
    _param_bounds: list
    _bounds_curr: list
    _bounds_volt: list

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self._settings.temp / elementary_charge

    @staticmethod
    def calc_error(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        return calculate_error_rae(y_pred, y_true)

    def __init__(self, settings_dev: SettingsDEV) -> None:
        """Class for emulating an electrical device"""
        self._settings = settings_dev
        self._init_class()

    def _check_right_param_format(self, new_list: list=()) -> bool:
        """Function for checking right keys of registered module and settings are equal"""
        keys_set = sorted([key for key in self._settings.dev_value.keys()])
        keys_bnd = sorted(new_list)
        keys_ref = sorted(self._type_device[self._settings.type]['param'])
        check = keys_set == keys_ref if len(new_list) == 0 else keys_bnd == keys_ref
        return check

    def register_device(self, short_label: str, description: str, param: list, func_equa, func_fit, func_reg) -> None:
        """Function for registering an electrical device to library
        :param short_label:     String with short label of the device (e.g. 'R')
        :param description:     Short description of device type (e.g. 'Resistor')
        :param param:           List with model parameter names
        :param func_equa:       Function to calculate the current output using normal equation
        :param func_fit:        Function to calculate the voltage output using in polynom fitting
        :param func_reg:        Function to calculate the voltage output using in regression
        """
        self._logger.debug(f"Registering device: {short_label} ({description}) with params: {param}")
        module = {short_label: {'desp': description, 'param': param,
                                'equa': func_equa, 'reg': func_reg, 'fit': func_fit}}
        self._type_device.update(module)

    def _init_class(self) -> None:
        self._logger = getLogger(__name__)
        self._type_device = dict()
        self._fit_done = False
        self._fit_options = [1, 1001]
        self._bounds_curr = [-14, -3]
        self._bounds_volt = [0.0, +6]

    def _extract_iv_curve_with_regression(self, params_dev: dict, bounds_voltage: list, bounds_current: list, mode_fit: int=0) -> dict:
        """Function for getting the I-V curve using electrical device description with regression
        Args:
            params_dev:             List with parameters from device
            bounds_voltage:         List for voltage limitation / range
            bounds_current:         List for current limitation / range
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
        Returns:
            Dictionary with two numpy arrays with current ['I'] and voltage ['V'] from device
        """
        u_path = np.zeros((1,), dtype=float)
        i_path = np.zeros((1,), dtype=float)

        if self._settings.type in self._type_device[self._settings.type]:
            device_method = self._type_device[self._settings.type]['func_reg']
            self._logger.debug(f"Apply regression function for getting device transfer function of {self._settings.type}")
            match mode_fit:
                case 0:
                    self._logger.debug("Generate I-V regression for given current boundries (positive range)")
                    i_path = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    u_path = -device_method(i_path, np.zeros(i_path.shape), params_dev)
                case 1:
                    self._logger.debug("Generate I-V regression for given current boundries (Full range)")
                    i_pathp = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    i_path = np.concatenate((-np.flipud(i_pathp[1:]), i_pathp), axis=0)
                    u_path = -device_method(i_path, np.zeros(i_path.shape), params_dev)
                case 2:
                    self._logger.debug("Generate I-V regression for given current boundries (Symmetric, Pos. mirrored)")
                    i_pathn = -np.logspace(bounds_current[1], bounds_current[0], self._fit_options[1], endpoint=True)
                    u_pathn = device_method(-i_pathn, np.zeros(i_pathn.shape), params_dev)
                    # --- Concatenate arrays
                    i_path = np.concatenate((i_pathn, -np.flipud(i_pathn)[1:]), axis=0)
                    u_path = np.concatenate((u_pathn, -np.flipud(u_pathn)[1:]), axis=0)
                case 3:
                    self._logger.debug("Generate I-V regression for given current boundries (Symmetric, Neg. mirrored)")
                    i_pathp = np.logspace(bounds_current[0], bounds_current[1], self._fit_options[1], endpoint=True)
                    u_pathp = device_method(i_pathp, np.zeros(i_pathp.shape), params_dev)
                    # --- Concatenate arrays
                    i_path = np.concatenate((-np.flipud(i_pathp)[1:], i_pathp), axis=0)
                    u_path = np.concatenate((-np.flipud(u_pathp)[1:], i_pathp), axis=0)
        else:
            self._logger.debug("Using normal device equation for getting I-V-behaviour")
            u_path = np.linspace(bounds_voltage[0], bounds_voltage[1], self._fit_options[1], endpoint=True)
            i_path = self.get_current(u_path, 0.0)

        # --- Limiting with voltage boundaries
        self._logger.debug("Truncating the electrical signals to desired bounds")
        x_start = int(np.argwhere(u_path >= bounds_voltage[0])[0])
        x_stop = int(np.argwhere(u_path >= bounds_voltage[1])[0])
        return {'I': i_path[x_start:x_stop], 'V': u_path[x_start:x_stop]}

    def _do_regression(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of the device with regression
        :param u_inp:           Positive input voltage [V]
        :param u_inn:           Negative input voltage [V]
        :return:                Corresponding current signal
        """
        if self._check_right_param_format():
            du = u_inp - u_inn
            if isinstance(du, float):
                du = list()
                du.append(u_inp - u_inn)

            # --- Start Conditions
            bounds = [10 ** self._bounds_curr[0], 10 ** self._bounds_curr[1]]
            y_initial = 2 * bounds[0]

            # --- Run optimization
            iout = list()
            self._logger.debug(f"Start regression of device: {self._settings.type}")
            for idx, u_sample in enumerate(du):
                sign_pos = u_sample >= 0.0
                y_start = y_initial if idx == 0 else abs(iout[-1])
                result = least_squares(
                    self._type_device[self._settings.type]['reg'],
                    y_start,
                    jac='3-point',
                    bounds=(bounds[0], bounds[1]),
                    args=(abs(u_sample), self._settings.dev_value)
                )
                iout.append(result.x[0] if sign_pos else -result.x[0])
            return np.array(iout, dtype=float)
        else:
            raise KeyError("Parameter keys are not identical")

    def _test_fit_option(self, params_used: dict, methods_compare: list, u_inn: float=0.0,
                         do_test: bool=False, do_plot: bool=True, plot_title_prefix: str='', path2save: str='') -> float:
        """Function for testing and plotting the comparison
        :param params_used:             Dictionary with device parameters
        :param methods_compare:         List with string labels of used method
        :param u_inn:                   Floating value with reference voltage [V]
        :param do_test:                 Performing a test
        :param do_plot:                 Plotting the results of regression and polynom fitting
        :param plot_title_prefix:       String for plot title as prefix
        :param path2save:               String with path to save the figure
        :return:                        Floating with error value [-1.0 = not available]
        """
        if do_test:
            self._logger.debug(f"Make IV comparison: {methods_compare[0]} vs. {methods_compare[1]}")
            u_poly = np.linspace(self._bounds_volt[0], self._bounds_volt[1], self._fit_options[1], endpoint=True)
            i_poly = self._type_device[self._settings.type](u_poly, u_inn)
            i_test = self._do_regression(u_poly, u_inn)
            error = self.calc_error(i_poly, i_test)

            self._plot_transfer_function_comparison(
                u_transfer=u_poly,
                i_dev0=i_poly,
                i_dev1=i_test,
                method_types=methods_compare,
                metric=['1e3 * RAE', 1e3 * error],
                title_prefix=plot_title_prefix,
                path2save=path2save,
                show_plot=do_plot
            )
        else:
            self._logger.debug(f"Make no IV comparison")
            error = -1.0
        return error

    def _get_params_from_curve_fitting(self, voltage: np.ndarray, current: np.ndarray, bounds_params: dict={},
                                       mode_fit: int=0, do_test: bool=False, do_plot: bool=True,
                                       plot_title_prefix: str='', path2save: str='') -> [np.ndarray, float]:
        """Function to extract the params of electrical device behaviour with curve fitting
        Args:
            voltage:                Numpy array with voltage from measurement
            current:                Numpy array with current from measurement
            bounds_params:          Dictionary with param bounds
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
            do_test:                Performing a test
            do_plot:                Plotting the results of regression and polynom fitting
            plot_title_prefix:      String for plot title as prefix
            path2save:              String with path to save the figure
        Returns:
            List with device parameter and floating value with Relative Squared Error
        """
        if len(bounds_params):
            self.declare_param_bounds(bounds_params)

        params = self.get_params_from_fitting_data(
            voltage=voltage,
            current=current,
            use_param_bounds=len(bounds_params) > 0
        )
        i_path, u_path = self._extract_iv_curve_with_regression(
            params_dev=params,
            bounds_voltage=self._bounds_volt,
            bounds_current=self._bounds_curr,
            mode_fit=mode_fit
        )
        self._logger.debug(f"Start curve fitting of device: {self._settings.type}")
        error = self._test_fit_option(
            params_used=params,
            methods_compare=['Curve fitting', 'Regression'],
            do_test=do_test,
            do_plot=do_plot,
            plot_title_prefix=plot_title_prefix,
            path2save=path2save
        )
        return [params, error]

    def _get_params_for_polynomfit(self, mode_fit: int=0,
                                   do_test: bool=False, do_plot: bool=False,
                                   plot_title_prefix: str='', path2save: str='') -> [np.ndarray, float]:
        """Function to extract the params of electrical device behaviour with polynom fit function
        Args:
            do_test:                Performing a test
            mode_fit:               Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
            do_plot:                Plotting the results of regression and polynom fitting
            plot_title_prefix:      String for plot title as prefix
            path2save:              String with path to save the figure
        Returns:
            List with polynom fit parameter and floating value with Relative Squared Error
        """
        params_used = self._settings.dev_value
        i_path, u_path = self._extract_iv_curve_with_regression(
            params_dev=params_used,
            bounds_voltage=self._bounds_volt,
            bounds_current=self._bounds_curr,
            mode_fit=mode_fit
        )
        self._logger.debug(f"Start polynom fitting of device: {self._settings.type}")
        params = np.polyfit(x=u_path, y=i_path, deg=self._fit_options[0])
        error = self._test_fit_option(
            params_used=params_used,
            methods_compare=['Poly. fitting', 'Regression'],
            u_inn=0.0,
            do_test=do_test,
            do_plot=do_plot,
            plot_title_prefix=plot_title_prefix,
            path2save=path2save
        )

        return [params, error]

    def _plot_transfer_function_comparison(self, u_transfer: np.ndarray, i_dev0: np.ndarray, i_dev1: np.ndarray,
                                           method_types: list, metric: list=(), title_prefix: str='',
                                           path2save: str='', show_plot: bool=False) -> None:
        """Plotting the transfer function of electrical device for comparison
        Args:
            u_transfer:     Numpy array with voltage from polynom fit (input)
            i_dev0:         Numpy array of current response from first method
            i_dev1:         Numpy array of current response from second method
            method_types:   List with string labels of used methods
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
        axs[0].semilogy(u_transfer, 1e6 * np.abs(i_dev0), 'r', marker='.', markersize=2, label=method_types[0])
        axs[0].grid()
        axs[0].set_ylabel(r'Current $\log_{10}(I_F)$ / µA')

        axs[1].plot(u_transfer, 1e6 * i_dev0, 'r', marker='.', markersize=2, label=method_types[0])
        axs[1].grid()
        axs[1].set_ylabel(r'Current $I_F$ / µA')
        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')

        if not np.array_equal(i_dev1, np.zeros_like(i_dev0)):
            axs[0].semilogy(u_transfer, 1e6 * np.abs(i_dev1), 'k', marker='.', markersize=2, label=method_types[1])
            axs[1].plot(u_transfer, 1e6 * i_dev1, 'k', marker='.', markersize=2, label=method_types[1])
            axs[1].legend()

        # --- Add figure title
        if not len(metric) == 0:
            axs[0].set_title(title_prefix + f"{metric[0]} = {metric[1]:.4f} @ N_Poly = {self._fit_options[0]}")

        if path2save:
            save_figure(plt, path2save, 'device_iv_charac.svg')
        if show_plot:
            plt.show(block=True)

    def _find_best_poly_order(self, order_start: int, order_stop: int,
                              show_plots: bool=False, mode_fit: int=0) -> None:
        """Finding the best polynomial order for fitting
        Args:
            order_start:    Integer value with starting order number
            order_stop:     Integer value with stopping order number
            show_plots:     Showing plots of each run
            mode_fit:       Fit Range Mode [0: Full Pos., 1: Full +/-, 2: Take Pos., Mirror Neg., 3: Take Neg., Mirror Pos.]
        Returns:
            None
        """
        print("\n=====================================================")
        print("Searching the best polynom order with minimal error")
        print("=====================================================")
        order_search = [idx for idx in range(order_start, order_stop+1)]
        error_search = []
        for idx, order in enumerate(order_search):
            self.change_options_fit(order, self._fit_options[1])
            error = self._get_params_for_polynomfit(
                do_test=True, do_plot=show_plots,
                mode_fit=mode_fit
            )
            error_search.append(error)
            print(f"#{idx:02d}: order = {order:02d} --> Error = {error}")

        # --- Finding best order
        error_search = np.array(error_search)
        xmin = np.argwhere(error_search == error_search.min()).flatten()
        print(f"\nBest solution: Order = {np.array(order_search)[xmin]} with an error of {error_search[xmin]}!")

    def plot_polyfit_tranfer_function(self, find_best_order: bool=False, show_plots: bool=True,
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
        if not find_best_order:
            self._get_params_for_polynomfit(
                do_test=True, do_plot=show_plots,
                mode_fit=mode_fit
            )
        else:
            self._find_best_poly_order(
                order_start=order_start, order_stop=order_stop,
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
            print(f"\t#{idx:03d}: {type} = {self._type_device[type]['desp']}")

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
        Returns:
            Corresponding current response
        """
        method = [method for method in self._type_device.keys() if method == self._settings.type]
        if len(method) and self._check_right_param_format():
            iout = self._type_device[self._settings.type]['equa'](u_top, u_bot, self._settings.dev_value)
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

    def _build_param_bounds_list(self, param_bounds: dict) -> list:
        """Building the list using the arg names of device parameters with external dict of parameter limits"""
        if self._check_right_param_format([key for key in param_bounds.keys()]):
            arg_names = [key for key in self._settings.dev_value.keys()]
            param_bounds_list = [[param_bounds[key][0] for key in arg_names], [param_bounds[key][1] for key in arg_names]]
            return param_bounds_list
        else:
            raise KeyError(f"Wrong parameter names! Use: {self._type_device[self._settings]['param']}")

    def declare_param_bounds(self, param_bounds: dict) -> list:
        """Function for building the param bounds used in curve fitting
        :param param_bounds:    Dictionary with {parameter name, [min, max]}
        """
        self._param_bounds = self._build_param_bounds_list(param_bounds)
        return self._param_bounds

    def get_params_from_fitting_data(self, voltage: np.ndarray, current: np.ndarray, use_param_bounds: bool) -> dict:
        """Function to extract the model parameters from fitting the measurement to model
        :param voltage:         Numpy array with voltage signal from IV-measurement [V]
        :param current:         Numpy array with current signal from IV-measurement [A]
        :param use_param_bounds:Boolean for using param bounds in curve fitting (declaration with declare_param_bounds())
        :return:                Dictionary with model parameters
        """
        if self._check_right_param_format():
            arg_names = [key for key in self._settings.dev_value.keys()]
            self._logger.debug(f"Getting the model parameters: {arg_names}")
            bounds_used = self._param_bounds if use_param_bounds else [[-np.inf for arg in arg_names], [np.inf for arg in arg_names]]

            params, coinv = curve_fit(
                f=self._type_device[self._settings.type]['fit'],
                ydata=voltage,
                xdata=current,
                bounds=bounds_used,
                absolute_sigma=True,
                check_finite=True,
                full_output=False
            )
            self._logger.debug(f"Coinv values of curve fitting: {coinv}")
            return dict(zip(arg_names, params))
        else:
            raise KeyError("Wrong Key List with Parameters")

    def get_voltage_from_fitting(self, current: np.ndarray, params: dict) -> np.ndarray:
        """"""
        return self._type_device[self._settings.type]['fit'](current, **params)

    def get_current_from_fitting(self, voltage: np.ndarray, params: dict) -> np.ndarray:
        """"""
        return self._type_device[self._settings.type]['equa'](voltage, np.zeros_like(voltage), params)

    def check_value_range_violation(self, signal: np.ndarray | float, mode_voltage: bool=True) -> bool:
        """Checking differential input stream has a violation against given range
        :param signal:          Numpy array with applied voltage difference [V]
        :param mode_voltage:    Boolean if input signal is voltage [True] or current [False]
        :return:                Boolean if warning violation is available
        """
        range_list = self._bounds_volt if mode_voltage else self._bounds_curr
        violation_dwn = np.count_nonzero(signal < range_list[0], axis=0)
        violation_up = np.count_nonzero(signal > range_list[1], axis=0)

        if violation_up or violation_dwn:
            val = signal.min if violation_dwn else signal.max
            limit = range_list[0] if violation_dwn else range_list[1]
            addon = f'(Upper limit)' if not violation_dwn else '(Downer limit)'
            self._logger.warn(f"Voltage Range Violation {addon}! With {val} of {limit} ---")
        return violation_up or violation_dwn


def generate_test_signal(t_end: float, fs: float, upp: list, fsig: list, uoff: float=0.0) -> [np.ndarray, np.ndarray]:
    """Generating a signal for testing
    Args:
        t_end:      End of simulation
        fs:         Sampling rate
        upp:        List with amplitude values
        fsig:       List with corresponding frequency
        uoff:       Offset voltage
    Returns:
        List with two numpy arrays (time, voltage signal)
    """
    t0 = np.linspace(start=0, stop=t_end, num=int(t_end * fs)+1, endpoint=True)
    uinp = np.zeros(t0.shape) + uoff
    for upp0, fsig0 in zip(upp, fsig):
        uinp += upp0 * np.sin(2 * np.pi * t0 * fsig0)
    return t0, uinp


def plot_test_results(time: np.ndarray, u_in: np.ndarray, i_in: np.ndarray,
                      mode_current_input: bool, do_ylog: bool=False, plot_gray: bool=False,
                      path2save: str='', show_plot: bool=False) -> None:
    """Function for plotting transient signal and I-V curve of the used electrical device
    Args:
        time:       Numpy array with time information
        u_in:       Numpy array with input voltage (mode_current_input = False) or output voltage (True)
        i_in:       Numpy array with output current (mode_current_input = False) or input current (True)
        mode_current_input: Bool decision for selecting right source and sink value
        do_ylog:    Plotting the current in the I-V-curve normal (False) or logarithmic (True)
        plot_gray:  Plotting the response of device in red dashed (False) or gray dashed (True)
        path2save:  Path for saving the plot
        show_plot:  Showing and blocking the plot
    Returns:
        None
    """
    scale_i, units_i = scale_auto_value(i_in)
    scale_u, units_u = scale_auto_value(u_in)
    scale_t, units_t = scale_auto_value(time)

    signalx = scale_i * i_in if mode_current_input else scale_u * u_in
    signaly = scale_u * u_in if mode_current_input else scale_i * i_in
    label_axisx = f'Voltage U_x [{units_u}V]' if mode_current_input else f'Current I_x [{units_i}A]'
    label_axisy = f'Current I_x [{units_i}A]' if mode_current_input else f'Voltage U_x [{units_u}V]'
    label_legx = 'i_in' if mode_current_input else 'u_in'
    label_legy = 'u_out' if mode_current_input else 'i_out'

    # --- Plotting: Transient signals
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(scale_t * time[0], scale_t * time[-1])
    twin1 = axs[0].twinx()
    a = axs[0].plot(scale_t * time, signalx, 'k', label=label_legx)
    axs[0].set_ylabel(label_axisy)
    axs[0].set_xlabel(f'Time t [{units_t}s]')
    if plot_gray:
        b = twin1.plot(scale_t * time, signaly, linestyle='dashed', color=[0.5, 0.5, 0.5], label=label_legy)
    else:
        b = twin1.plot(scale_t * time, signaly, 'r--', label=label_legy)
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
    if path2save:
        save_figure(plt, path2save, 'test_signal')
    if show_plot:
        plt.show(block=True)
