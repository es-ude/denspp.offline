import numpy as np
from .dev_handler import ElectricalLoadHandler, SettingsDEV
from .dev_noise import ProcessNoise, SettingsNoise


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


class ElectricalLoad(ProcessNoise, ElectricalLoadHandler):
    _settings: SettingsDEV
    _poly_fit: np.ndarray
    _fit_options: list
    _curve_fit: np.ndarray
    _approx_fit: np.ndarray

    _type_device: dict
    _type_string: dict
    _type_params: dict
    _type_func2reg: dict
    _type_func2cur: dict
    _type_func2app: dict

    _bounds_curr: list
    _bounds_volt: list
    _params_used: list

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._init_class()

        self._settings = settings_dev
        self._type_device = self.__init_dev()
        self._type_string = self.__init_dev_string()
        self._type_params = self.__init_params()
        self._type_func2reg = self.__init_func2reg()
        self._type_func2cur = self.__init_func2curve()
        self._fit_options = [6, 1001]

    def __init_dev(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': self._resistor}
        dev_type.update({'Ds': self._diode_single, 'Dd': self._diode_antiparallel})
        dev_type.update({'DSs': self._diode_single, 'DSd': self._diode_antiparallel})
        dev_type.update({'RDs': self._resistive_schottky_single, 'RDd': self._resistive_schottky_antiparallel})
        return dev_type

    @staticmethod
    def __init_dev_string() -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': 'Resistor'}
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
        """Initialization of Device Parameters"""
        params_dict = {}
        params_dict.update({'R': [self._settings.dev_value]})
        params_dict.update({'Ds': [1e-12, 1.4, 0.7], 'Dd': [1e-12, 1.4, 0.7]})
        params_dict.update({'DSs': [1e-12, 1.4, 0.2], 'DSd': [1e-12, 1.4, 0.2]})
        params_dict.update({'RDs': [1e-12, 2.8, 0.1, self._settings.dev_value]})
        params_dict.update({'RDd': [1e-12, 2.8, 0.1, self._settings.dev_value]})
        return params_dict

    def _resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical resistor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        self._params_used = self._type_params[self._settings.type]

        du = u_inp - u_inn
        i_out = du / self._params_used[0]
        if self._settings.noise_en:
            i_out += self.gen_noise_awgn_curr(du.size, self._params_used[0])
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
            i_out += self.gen_noise_awgn_pwr(du.size)
        return i_out

    def _diode_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a single-ended pn-diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        self._params_used = self._type_params[self._settings.type]
        return self._func2equa_diode(self._params_used, u_inp, u_inn)

    def _diode_antiparallel(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a pn-diode (anti-parallel)
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        self._params_used = self._type_params[self._settings.type]
        i_pos = self._func2equa_diode(self._params_used, u_inp, u_inn)
        i_neg = self._func2equa_diode(self._params_used, -u_inp, -u_inn)
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

    @staticmethod
    def _func2curve_resistive_diode(i_path: np.ndarray, a, b, c, d) -> np.ndarray:
        """Function for performing curve fitting for resistive diode behaviour"""
        return a + b * i_path + c * np.log(d * i_path + 1)

    def _resistive_schottky_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float, mode_fitting: int=1) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and single-side schottky diode
        Args:
            u_inp:          Positive input voltage [V]
            u_inn:          Negative input voltage [V]
            mode_fitting:   Mode selection for fitting [0: None, 1: Polynomial, 2: Curve, 3: Approx]
        Returns:
            Corresponding current signal
        """
        self._params_used = self._type_params[self._settings.type]
        self._bounds_volt = [1.0, self._bounds_volt[1]]
        self._bounds_curr = [-15, -2]

        du = u_inp - u_inn
        if isinstance(du, float):
            du = np.zeros(shape=(1,))
            du[0] = u_inp - u_inn

        if mode_fitting == 1:
            # Polynomial fitting
            if self._poly_fit.size == 1:
                self._get_params_polyfit(self._params_used)
            i_fit = np.polyval(self._poly_fit, du)
        elif mode_fitting == 2:
            # Curve fitting
            if self._curve_fit.size == 1:
                self._get_params_curve_fit(self._params_used, mode_fit=0)
            i_fit = self._func2curve_resistive_diode(
                i_path=du,
                a=self._curve_fit[0],
                b=self._curve_fit[1],
                c=self._curve_fit[2],
                d=self._curve_fit[3]
            )
        else:
            # Regression
            i_fit = self._do_regression(u_inp, u_inn, self._bounds_volt, self._bounds_curr)

        # --- Checking if voltage limits or current limits are reached
        xpos_uth = np.argwhere(du < 0.75).flatten()
        if not xpos_uth.size == 0:
            i_fit[xpos_uth] = self._params_used[0]
        xpos_i0 = np.argwhere(i_fit < 0.0).flatten()
        if not xpos_i0.size == 0:
            i_fit[xpos_i0] = self._params_used[0]

        # --- Adding noise
        if self._settings.noise_en:
            i_fit += self.gen_noise_awgn_curr(du.size, self._params_used[-1])
        return i_fit

    def _resistive_schottky_antiparallel(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and double-side schottky diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_fit = self._resistive_schottky_single(np.abs(du), 0.0)

        xpos_neg = np.argwhere(du < 0.0).flatten()
        if not xpos_neg.size == 0:
            i_fit[xpos_neg] = (-1) * i_fit[xpos_neg]

        return i_fit
