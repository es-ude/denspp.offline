import numpy as np
from scipy.integrate import cumtrapz
from package.analog.dev_load import ElectricalLoad_Handler, SettingsDEV, _generate_signal, _plot_test_results
from package.analog.dev_noise import ProcessNoise, SettingsNoise


RecommendedSettingsDEV = SettingsDEV(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    dev_value=100e3,
    temp=300
)

RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
)


class ElectricalLoad(ProcessNoise, ElectricalLoad_Handler):
    _settings: SettingsDEV
    _poly_fit: np.ndarray
    _fit_options: list
    _curve_fit: np.ndarray
    _approx_fit: np.ndarray
    _bounds_current: list
    _bounds_voltage: list
    _type_device: dict
    _type_string: dict
    _type_params: dict
    _type_func2reg: dict
    _type_func2cur: dict
    _type_func2app: dict

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev
        self._init_class()
        self._type_device = self.__init_dev()
        self._type_string = self.__init_dev_string()
        self._type_params = self.__init_params()
        self._type_func2reg = self.__init_func2reg()
        self._type_func2cur = self.__init_func2curve()
        self._fit_options = [6, 1001]

    def __init_dev(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': self._resistor, 'C': self._capacitor, 'L': self._inductor}
        dev_type.update({'Ds': self._diode_single, 'Dd': self._diode_antiparallel})
        dev_type.update({'DSs': self._diode_single, 'DSd': self._diode_antiparallel})
        dev_type.update({'RDs': self._resistive_schottky_single, 'RDd': self._resistive_schottky_antiparallel})
        return dev_type

    def __init_dev_string(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': 'Resistor', 'C': 'Capacitor', 'L': 'Inductor'}
        dev_type.update({'Ds': 'pn-Diode (single)', 'Dd': 'pn-Diode (anti-parallel)'})
        dev_type.update({'DSs': 'Schottky diode (single)', 'DSd': 'Schottky diode (anti-parallel)'})
        dev_type.update({'RDs': 'Resistive schottky diode (single)', 'RDd': 'Resistive schottky diode (anti-parallel)'})
        return dev_type

    def __init_func2reg(self) -> dict:
        """Initialization of functions for regression"""
        func_type = {'RDs': self._func2reg_resistive_diode}
        func_type.update({'RDd': self._func2reg_resistive_diode})
        return func_type

    def __init_func2curve(self) -> dict:
        """Initialization of functions for curve fitting"""
        func_type = {'RDs': self._func2curve_resistive_diode}
        func_type.update({'RDd': self._func2curve_resistive_diode})
        return func_type

    def __init_func2approx(self) -> dict:
        """Initialization of functions for approximation fitting"""
        func_type = {'RDs': [self._diode_single, self._resistor]}
        func_type.update({'RDd': [self._diode_single, self._resistor]})
        return func_type

    def __init_params(self) -> dict:
        """"""
        params_dict = {}
        params_dict.update({'R': [self._settings.dev_value], 'C': [self._settings.dev_value], 'L': [self._settings.dev_value]})
        params_dict.update({'Ds': [1e-12, 1.4, 0.7], 'Dd': [1e-12, 1.4, 0.7]})
        params_dict.update({'DSs': [1e-12, 1.4, 0.2], 'DSd': [1e-12, 1.4, 0.2]})
        params_dict.update({'RDs': [0.1e-12, 2.8, 0.1, self._settings.dev_value]})
        params_dict.update({'RDd': [0.1e-12, 2.8, 0.1, self._settings.dev_value]})
        return params_dict

    def _resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical resistor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params[self._settings.type]
        du = u_inp - u_inn
        i_out = du / params[0]
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_curr(du.size, params[0])
        return i_out

    def _capacitor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical capacitor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params[self._settings.type]
        du = u_inp - u_inn
        i_out = np.zeros(du.shape)
        i_out[1:] = params[0] * np.diff(du) * self._settings.fs_ana
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _inductor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical inductor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params[self._settings.type]
        du = u_inp - u_inn
        i_out = cumtrapz(du, dx=1 / self._settings.fs_ana, initial=0) / params[0]
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _func2equa_diode(self, params: list, u_inp: np.ndarray, u_inn: np.ndarray) -> np.ndarray:
        """Function for getting the current relationship from the voltage input for a diode
        Args:
            params:     List with device parameter [Is0, n0, Uth]
            u_inp:      Positive input voltage [V]
            u_inn:      Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = params[0] * np.exp((du - params[2]) / (params[1] * self.temperature_voltage))
        # --- Current limitation
        xpos = np.argwhere(i_out >= 1.0).flatten()
        if not xpos.size == 0:
            i_out[xpos] = 1.0

        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _diode_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a single-ended pn-diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params[self._settings.type]
        return self._func2equa_diode(params, u_inp, u_inn)

    def _diode_antiparallel(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a pn-diode (anti-parallel)
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params[self._settings.type]
        i_pos = self._func2equa_diode(params, u_inp, u_inn)
        i_neg = self._func2equa_diode(params, -u_inp, -u_inn)
        return i_pos + i_neg

    def _func2reg_resistive_diode(self, i_path: np.ndarray, params: list, xd: np.ndarray) -> np.ndarray:
        """Function for do least_squared regression
        Args:
            i_path: Input current sample (--> output value)
            params: Device Parameter [Is0, n, Uth, Rs]
            xd:     Input difference voltage sample
        Returns:
            Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        v1 = params[2] + params[1] * self.temperature_voltage * np.log(i_path / params[0] + 1)
        v3 = params[3] * i_path
        return xd - v1 - v3

    def _func2curve_resistive_diode(self, i_path: np.ndarray, a, b, c, d) -> np.ndarray:
        """Function for performing curve fitting for resistive diode behaviour"""
        return a + b * i_path + c * np.log(d * i_path + 1)

    def _resistive_schottky_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float,
                                   mode_fitting=1) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and single-side schottky diode
        Args:
            u_inp:          Positive input voltage [V]
            u_inn:          Negative input voltage [V]
            mode_fitting:   Mode selection for fitting [0: None, 1: Polynomial, 2: Curve, 3: Approx]
        Returns:
            Corresponding current signal
        """
        params = self._type_params['RDs']
        bounds_voltage = [0.0, self._bounds_voltage[1]]

        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)

        if mode_fitting == 1:
            # --- Fitting
            if self._poly_fit.size == 1:
                self._get_params_polyfit(params, bounds_voltage, self._bounds_current)
            i_fit = np.polyval(self._poly_fit, du)
        elif mode_fitting == 2:
            # --- Curve fitting
            if self._curve_fit.size == 1:
                self._get_params_curve_fit(params, bounds_voltage, self._bounds_current)
            i_fit = self._func2curve_resistive_diode(du, self._curve_fit[0], self._curve_fit[1],
                                                     self._curve_fit[2], self._curve_fit[3])
        else:
            # --- Regression
            i_fit = self._do_regression(u_inp, u_inn, params, self._bounds_current)

        i_fit[i_fit < 0.0] = params[0]
        if self._settings.noise_en:
            i_fit += self._gen_noise_awgn_pwr(du.size)
        return i_fit

    def _resistive_schottky_antiparallel(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and double-side schottky diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self._type_params['RDd']
        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)

        if self._poly_fit.size == 1:
            self._get_params_polyfit(params, self._bounds_voltage, self._bounds_current)
        i_fit = np.polyval(self._poly_fit, du)

        if self._settings.noise_en:
            i_fit += self._gen_noise_awgn_pwr(du.size)
        return i_fit

    def plot_fit_curve(self, find_best_order=False, show_plots=False) -> None:
        """Plotting the output of the polynom fit function
        Args:
            find_best_order:    Find the best poly.-fit order
            show_plots:         Showing plots of each run
        Returns:
            None
        """
        params = self._type_params[self._settings.type]
        if not find_best_order:
            self._get_params_polyfit(
                params_dev=params,
                bounds_voltage=self._bounds_voltage,
                bounds_current=self._bounds_current,
                do_test=True,
                mode_fit=0
            )
        else:
            self._find_best_poly_order(3, 18,
                                       bounds_voltage=self._bounds_voltage, params_dev=params,
                                       mode_fitting=0, show_plots=show_plots)


# --------------------- TEST CASE ---------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    settings = SettingsDEV(
        type='RDs',
        fs_ana=500e3,
        noise_en=False,
        para_en=False,
        dev_value=100e3,
        temp=300
    )

    # --- Declaration of input
    do_ylog = False
    t_end = 0.5e-3
    u_off = 2.6

    t0, uinp = _generate_signal(0.5e-3, settings.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 0.0)
    uinp = uinp + u_off
    uinn = 0.0

    # --- Model declaration
    plt.close('all')
    dev = ElectricalLoad(settings)
    dev.print_types()

    # --- Plotting: I-V curve
    print("\nPlotting I-V curve")
    dev.change_boundary_voltage(1.33, 6.0)
    dev.plot_fit_curve()

    # --- Plotting: Current response
    print("\nPlotting transient response")
    iout = dev.get_current(uinp, uinn)
    _plot_test_results(t0, uinp - uinn, iout, False, do_ylog)

    # --- Plotting: Voltage response
    uout = dev.get_voltage(iout, uinn, u_off, 1e-2)
    _plot_test_results(t0, uout+uinn, iout, True, do_ylog)
    plt.show()


