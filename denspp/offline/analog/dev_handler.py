import numpy as np
from tqdm import tqdm
from inspect import getfullargspec
from dataclasses import dataclass
from logging import getLogger
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import least_squares, curve_fit

from denspp.offline.plot_helper import scale_auto_value, save_figure
from denspp.offline.analog.iv_polyfit import PolyfitIV
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise
from denspp.offline.metric.data_numpy import calculate_error_rae, calculate_error_mse


@dataclass
class SettingsDevice:
    """Individual data class to configure an electrical device for simulation
    Attributes:
        type:       Type of electrical device ['R': resistor, 'RDs': Resistive diode (series), 'RDd': Resistive diode (antiparallel)]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        params_use: Dictionary with device parameters
        temp:       Temperature [K]
        use_poly:   Boolean for using polynom fit [True] of IV curve for transient data analysis instead of regression [False]
    """
    type:       str
    fs_ana:     float
    noise_en:   bool
    params_use: dict
    temp:       float
    use_poly:   bool

    @property
    def temperature_voltage(self) -> float:
        """Getting the Temperature voltage of electrical device"""
        return Boltzmann * self.temp / elementary_charge

    @property
    def dev_value(self) -> dict:
        """Getting the model device parameter for simulation (can be rewritten in new classes)"""
        return self.params_use


class ElectricalLoadHandler(ProcessNoise):
    _settings_device: SettingsDevice
    _settings_noise: SettingsNoise
    _type_device: dict
    # --- Fitting options
    _polynom_fitter: PolyfitIV
    # --- Electrical Signal limitations
    _param_bounds: list
    _bounds_curr: list
    _bounds_volt: list

    def __init__(self, settings_dev: SettingsDevice, settings_noise: SettingsNoise=RecommendedSettingsNoise) -> None:
        """Class for emulating an electrical device
        :param settings_dev:        Class for controlling the device simulation
        :param settings_noise:      Class for controlling the noise behaviour
        :return:                    None
        """
        ProcessNoise.__init__(self,
            settings=settings_noise,
            fs_ana=settings_dev.fs_ana
        )

        self._settings_device = settings_dev
        self._logger = getLogger(__name__)
        self._type_device = dict()
        self._polynom_fit = PolyfitIV(
            sampling_rate=settings_dev.fs_ana,
            en_noise=settings_dev.noise_en,
            settings_noise=settings_noise
        )

        self._num_points = 1001
        self._bounds_curr = [-14, -3]
        self._bounds_volt = [0.0, 5.0]

    @staticmethod
    def _calc_error(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        return calculate_error_rae(y_pred, y_true)

    def _check_right_param_format(self, new_list: list=()) -> bool:
        """Function for checking right keys of registered module and settings are equal"""
        keys_set = sorted([key for key in self._settings_device.dev_value.keys()])
        keys_bnd = sorted(new_list)
        keys_ref = sorted(self._type_device[self._settings_device.type]['param'])
        check = keys_set == keys_ref if len(new_list) == 0 else keys_bnd == keys_ref
        return check

    def _register_device(self, short_label: str, description: str, func_equa, func_fit, func_reg) -> None:
        """Function for registering an electrical device to library
        :param short_label:     String with short label of the device (e.g. 'R')
        :param description:     Short description of device type (e.g. 'Resistor')
        :param func_equa:       Function to calculate the current output using normal equation
        :param func_fit:        Function to calculate the voltage output using in polynom fitting
        :param func_reg:        Function to calculate the voltage output using in regression
        """
        param = getfullargspec(func_fit)[0][2:]
        self._logger.debug(f"Registering device: {short_label} ({description}) with params: {param}")
        module = {short_label: {'desp': description, 'param': param, 'equa': func_equa, 'reg': func_reg, 'fit': func_fit}}
        self._type_device.update(module)

    def _extract_iv_curve_with_polyfit(self, current: np.ndarray, voltage: np.ndarray, show_plots: bool=False,
                                       find_best_order: bool=False, order_range: list=(2, 18)) -> float:
        """Extracting the polynom fit parameters and plotting it compared to regression task
        :param current:             Numpy array with current values
        :param voltage:             Numpy array with voltage values
        :param find_best_order:     Find the best poly.-fit order
        :param order_range:         Range with Integer value for search (best polynom order)
        :param show_plots:          Showing plots of each run
        :return:                    Floating value with error
        """
        return self._polynom_fit.extract_polyfit_params(
            current=current,
            voltage=voltage,
            show_plots=show_plots,
            find_best_order=find_best_order,
            order_range=order_range
        )

    def _extract_iv_curve_with_regression(self, params_dev: dict) -> dict:
        """Function for getting the I-V curve using electrical device description with regression
        Args:
            params_dev:             Dictionary with parameters from device
        Returns:
            Dictionary with two numpy arrays with current ['I'] and voltage ['V'] from device
        """
        if self._settings_device.type in self._type_device[self._settings_device.type]:
            self._logger.debug(f"Apply regression function for getting device transfer function of {self._settings_device.type}")
            self._logger.debug("Generate I-V regression for given current boundries (Negative range)")
            u_path = np.logspace(start=self._bounds_volt[0], stop=self._bounds_volt[1], num=self._num_points, endpoint=True)
            i_path = self._do_regression(u_inp=u_path, u_inn=0.0, params=params_dev, disable_print=True)
        else:
            self._logger.debug("Using normal device equation for getting I-V-behaviour")
            u_path = np.linspace(self._bounds_volt[0], self._bounds_volt[1], self._num_points, endpoint=True)
            i_path = self._get_current_from_equation(u_path, 0.0, params_dev)

        # --- Limiting with voltage boundaries
        self._logger.debug("Truncating the electrical signals to desired bounds")
        x_start = int(np.argwhere(u_path >= self._bounds_volt[0])[0])
        x_stop = int(np.argwhere(u_path >= self._bounds_volt[1])[0])
        return {'I': i_path[x_start:x_stop], 'V': u_path[x_start:x_stop]}

    def _do_regression(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float,
                       params: dict=(), disable_print: bool=False) -> np.ndarray:
        """Performing the behaviour of the device with regression
        :param u_inp:           Positive input voltage [V]
        :param u_inn:           Negative input voltage [V]
        :param params:          Dictionary with model parameters
        :param disable_print:   Disabling the tqdm print
        :return:                Corresponding current signal
        """
        params_used = self._settings_device.dev_value if len(params) == 0 else params
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
            self._logger.debug(f"Start regression of device: {self._settings_device.type}")
            for idx, u_sample in enumerate(tqdm(du, desc="Regression Progress:", disable=disable_print)):
                sign_pos = u_sample >= 0.0
                y_start = y_initial if idx == 0 else abs(iout[-1])
                result = least_squares(
                    self._type_device[self._settings_device.type]['reg'],
                    y_start,
                    jac='3-point',
                    bounds=(bounds[0], bounds[1]),
                    args=(abs(u_sample), params_used)
                )
                iout.append(result.x[0] if sign_pos else -result.x[0])
            return np.array(iout, dtype=float)
        else:
            raise KeyError("Parameter keys are not identical")

    def _test_fit_option(self, voltage_test: np.ndarray, params_used: dict | list,
                         methods_compare: list, u_inn: float=0.0, plot_title: str='',
                         do_test: bool=False, do_plot: bool=True, path2save: str='') -> float:
        """Function for testing and plotting the comparison
        :param params_used:             Dictionary with device parameters
        :param methods_compare:         List with string labels of used method
        :param u_inn:                   Floating value with reference voltage [V]
        :param plot_title:              Title of plot
        :param do_test:                 Performing a test
        :param do_plot:                 Plotting the results of regression and polynom fitting
        :param path2save:               String with path to save the figure
        :return:                        Floating with error value [-1.0 = not available]
        """
        if do_test:
            self._logger.debug(f"Make IV comparison: {methods_compare[0]} vs. {methods_compare[1]}")
            i_test = self._do_regression(voltage_test, u_inn)
            if isinstance(params_used, dict):
                i_poly = [self._get_current_from_equation(voltage_test, u_inn, params_used)]
            else:
                i_poly = [self._polynom_fit.get_current(voltage_test, u_inn), u_inn + self._polynom_fitter.get_voltage(i_test)]

            error = self._calc_error(i_poly[0], i_test)
            plot_title_new = f"{plot_title}, 1e3* RAE = {error:.3f}" if plot_title else f"1e3* RAE = {error:.3f}"
            self._plot_transfer_function_comparison(
                u_transfer=voltage_test,
                i_dev0=i_poly,
                i_dev1=i_test,
                method_types=methods_compare,
                plot_title=plot_title_new,
                path2save=path2save,
                show_plot=do_plot
            )
        else:
            self._logger.debug(f"Make no IV comparison")
            error = -1.0
        return error

    def _get_params_from_curve_fitting(self, bounds_params: dict, do_test: bool=False,
                                       do_plot: bool=True, path2save: str='') -> [np.ndarray, float]:
        """Function to extract the params of electrical device behaviour with curve fitting
        Args:
            bounds_params:          Dictionary with param bounds
            do_test:                Performing a test
            do_plot:                Plotting the results of regression and polynom fitting
            path2save:              String with path to save the figure
        Returns:
            List with device parameter and floating value with Relative Squared Error
        """
        signals = self._extract_iv_curve_with_regression(params_dev=self._settings_device.dev_value)
        self._logger.debug(f"Start curve fitting of device: {self._settings_device.type}")
        params_ext = self.extract_params_curvefit(
            voltage=signals['V'],
            current=signals['I'],
            param_bounds=bounds_params
        )
        error = self._test_fit_option(
            voltage_test=signals['V'],
            params_used=params_ext,
            methods_compare=['Curve fitting (Remodeled)', 'Regression (Orig.)'],
            u_inn=0.0,
            plot_title='Model parameter extraction',
            do_test=do_test,
            do_plot=do_plot,
            path2save=path2save
        )
        return [params_ext, error]

    @staticmethod
    def _plot_transfer_function_comparison(u_transfer: np.ndarray, i_dev0: np.ndarray | list, i_dev1: np.ndarray,
                                           method_types: list, plot_title: str='',
                                           path2save: str='', show_plot: bool=False) -> None:
        """Plotting the transfer function of electrical device for comparison
        Args:
            u_transfer:     Numpy array with voltage from polynom fit (input)
            i_dev0:         Numpy array of current response from first method
            i_dev1:         Numpy array of current response from second method
            method_types:   List with string labels of used methods
            plot_title:     String with plot title
            path2save:      String with path to save the figure
            show_plot:      Showing and blocking the plots [Default: False]
        Returns:
            None
        """
        scaley, unity = scale_auto_value(i_dev1)
        plt.figure()
        plt.tight_layout()

        axs = list()
        axs.append(plt.subplot(2, 1, 1))
        axs.append(plt.subplot(2, 1, 2, sharex=axs[0]))
        axs[0].semilogy(u_transfer, scaley * np.abs(i_dev0[0]), 'r', marker='.', markersize=2, label=f"{method_types[0]} (Current)")
        axs[0].grid()
        axs[0].set_ylabel(r'Current $\log_{10}(I_F)$ / µA')

        axs[1].plot(u_transfer, scaley * i_dev0[0], 'r', marker='.', markersize=2, label=f"{method_types[0]} (Current)")
        axs[1].grid()
        axs[1].set_ylabel(fr'Current $I_F$ / {unity}A')
        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')

        if len(i_dev0) > 1:
            axs[0].semilogy(i_dev0[1], scaley * np.abs(i_dev1), 'g', marker='.', markersize=2, label=f"{method_types[0]} (Voltage)")
            axs[1].plot(i_dev0[1], scaley * i_dev1, 'g', marker='.', markersize=2, label=f"{method_types[0]} (Voltage)")

        if not np.array_equal(i_dev1, np.zeros_like(i_dev0[0])):
            axs[0].semilogy(u_transfer, scaley * np.abs(i_dev1), 'k', marker='.', markersize=2, label=method_types[1])
            axs[1].plot(u_transfer, scaley * i_dev1, 'k', marker='.', markersize=2, label=method_types[1])

        axs[1].legend()
        axs[0].set_title(plot_title)
        if path2save:
            save_figure(plt, path2save, 'device_iv_charac', ['svg'])
        if show_plot:
            plt.show(block=True)

    def get_type_list(self) -> dict:
        """Extracting the types as list
        :return:    Dictionaray with device types and corresponding description and parameter list
        """
        overview = dict()
        for key in self._type_device.keys():
            overview[key] = {'desp': self._type_device[key]['desp'], 'param': self._type_device[key]['param']}
        return overview

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

    def change_options_polyfit(self, poly_order: int, num_points_fit: int) -> None:
        """Redefining the options for polynom fitting of I-V behaviour of electrical devices
        Args:
            poly_order:     Order of the polynom fit
            num_points_fit: Exponential integer for downer voltage limit
        """
        self._num_points = num_points_fit
        self._polynom_fit.change_fit_settings(poly_order)

    def declare_param_bounds_curvefit(self, param_bounds: dict) -> list:
        """Function for building the param bounds used in curve fitting
        :param param_bounds:    Dictionary with {parameter name, [min, max]}
        :return:                List with bounds in right order
        """
        if self._check_right_param_format():
            arg_names_must = [key for key in self._settings_device.dev_value.keys()]
            arg_names_have = [key for key in param_bounds.keys()]
            self._param_bounds = [[param_bounds[key][0] if key in arg_names_have else -np.inf for key in arg_names_must],
                                  [param_bounds[key][1] if key in arg_names_have else np.inf for key in arg_names_must]]
            return self._param_bounds
        else:
            raise KeyError(f"Wrong parameter names! Use: {self._type_device[self._settings_device]['param']}")

    def _build_param_initial_guess(self) -> list:
        guess_values = list()
        for val_min, val_max in zip(self._param_bounds[0], self._param_bounds[1]):
            val_end = val_min + np.random.ranf(1) * (val_max-val_min)
            val_inf = 0.0
            guess_values.append(float(val_inf if np.isinf(val_min) or np.isinf(val_max) else val_end))
        return guess_values

    def extract_params_curvefit(self, voltage: np.ndarray, current: np.ndarray, param_bounds: dict) -> dict:
        """Function to extract the model parameters from fitting the measurement to model (curve fit)
        :param voltage:         Numpy array with voltage signal from IV-measurement [V]
        :param current:         Numpy array with current signal from IV-measurement [A]
        :param param_bounds:    Dictionary with parameter bounds {parameter name: [min, max], ...}
        :return:                Dictionary with model parameters
        """
        self.declare_param_bounds_curvefit(param_bounds)

        if self._check_right_param_format():
            arg_names = [key for key in self._settings_device.dev_value.keys()]
            self._logger.debug(f"Getting the model parameters: {arg_names}")
            params, coinv = curve_fit(
                f=self._type_device[self._settings_device.type]['fit'],
                ydata=voltage,
                xdata=current,
                bounds=self._param_bounds,
                p0=self._build_param_initial_guess(),
                method='trf'
            )
            self._logger.debug(f"Coinv values of curve fitting: {coinv}")
            return dict(zip(arg_names, params))
        else:
            raise KeyError("Wrong Key List with Parameters")

    def check_value_range_violation(self, signal: np.ndarray | float, mode_voltage: bool=True) -> bool:
        """Checking differential input stream has a violation against given range
        :param signal:          Numpy array with applied voltage difference [V]
        :param mode_voltage:    Boolean if input signal is voltage [True] or current [False]
        :return:                Boolean if warning violation is available
        """
        range_list = self._bounds_volt if mode_voltage else [0, 10**self._bounds_curr[1]]
        signal_used = signal if mode_voltage else np.abs(signal)
        violation_dwn = np.count_nonzero(signal_used < range_list[0], axis=0)
        violation_up = np.count_nonzero(signal_used > range_list[1], axis=0)

        if violation_up or violation_dwn:
            addon = f'(Upper limit)' if not violation_dwn else '(Downer limit)'
            self._logger.warn(f"Voltage Range Violation {addon}!")
        return bool(violation_up or violation_dwn)

    def _get_current_from_equation(self, voltage_pos: np.ndarray | float, voltage_neg: np.ndarray | float, params: dict) -> np.ndarray:
        return self._type_device[self._settings_device.type]['equa'](voltage_pos, voltage_neg, params)

    def _get_voltage_from_regression(self, current: np.ndarray, params: dict) -> np.ndarray:
        return -self._type_device[self._settings_device.type]['reg'](current, np.zeros_like(current), params)

    def _get_voltage_with_search(self, i_in: np.ndarray, u_inn: np.ndarray | float, start_value: float=0.0, start_step: float=1e-3, take_last_value: bool=True) -> np.ndarray:
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
        for i0 in tqdm(i_in, desc="Extract voltage value"):
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

    def get_voltage(self, current: np.ndarray, u_inn: float) -> np.ndarray:
        """Getting the voltage response from electrical device
        :param current: Applied current into device [A]
        :param u_inn:   Applied voltage on bottom electrode  [V]
        :returns:       Corresponding voltage response
        """
        method = [method for method in self._type_device.keys() if method == self._settings_device.type]
        if len(method) and self._check_right_param_format():
            if not self._settings_device.use_poly:
                return self._get_voltage_with_search(current, u_inn)
            else:
                if np.isnan(self._polynom_fit._fit_params_v2i).any():
                    vi = self._extract_iv_curve_with_regression(self._settings_device.dev_value)
                    error = self._extract_iv_curve_with_polyfit(
                        current=vi['I'],
                        voltage=vi['V']
                    )
                    self._logger.debug(f"Extracted IV curve for polyfitting with error of {error}")
                return self._polynom_fit.get_voltage(current) + u_inn
        else:
            if len(method):
                raise ValueError("Parameter 'type': Model not available - Please check!")
            else:
                ovr = self.get_type_list()
                raise ValueError(f"Parameter 'use_params': Wrong parameters selected - Please use {ovr[method[0]]['params']}!")

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        :param u_top:   Applied voltage on top electrode [V]
        :param u_bot:   Applied voltage on bottom electrode  [V]
        :returns:       Corresponding current response
        """
        method = [method for method in self._type_device.keys() if method == self._settings_device.type]
        if len(method) and self._check_right_param_format():
            if not self._settings_device.use_poly:
                return self._get_current_from_equation(u_top, u_bot, self._settings_device.dev_value)
            else:
                if np.isnan(self._polynom_fit._fit_params_v2i).any():
                    vi = self._extract_iv_curve_with_regression(self._settings_device.dev_value)
                    error = self._extract_iv_curve_with_polyfit(
                        current=vi['I'],
                        voltage=vi['V']
                    )
                    self._logger.debug(f"Extracted IV curve for polyfitting with error of {error}")
                return self._polynom_fit.get_current(u_top, u_bot)
        else:
            if len(method):
                raise ValueError("Parameter 'type': Model not available - Please check!")
            else:
                ovr = self.get_type_list()
                raise ValueError(
                    f"Parameter 'use_params': Wrong parameters selected - Please use {ovr[method[0]]['params']}!")

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
