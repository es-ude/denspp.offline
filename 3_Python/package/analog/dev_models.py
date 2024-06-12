import numpy as np
from scipy.integrate import cumtrapz
from package.analog.dev_load import (ElectricalLoad_Handler, SettingsDEV, _error_mse,
                                     _generate_signal, _plot_test_results)
from package.analog.dev_noise import ProcessNoise, SettingsNoise


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


class ElectricalLoad(ProcessNoise, ElectricalLoad_Handler):
    _settings: SettingsDEV
    _poly_fit: np.ndarray
    _type_device: dict
    _type_string: dict
    _type_func2reg: dict

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev
        self._poly_fit = np.zeros((1,), dtype=float)
        self._type_device = self.__init_dev()
        self._type_string = self.__init_dev_string()
        self._type_func2reg = self.__init_func()

        self._bounds_current = [-11, -2]
        self._bounds_voltage = [-20, +20]

    def __init_dev(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': self._resistor, 'C': self._capacitor, 'L': self._inductor}
        dev_type.update({'S': self._diode_schottky_single})
        dev_type.update({'Ds': self._diode_single_barrier, 'Dd': self._diode_double_barrier})
        dev_type.update({'RDs': self._resistive_schottky_single, 'RDd': self._resistive_schottky_double})
        return dev_type

    def __init_dev_string(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': 'Resistor', 'C': 'Capacitor', 'L': 'Inductor'}
        dev_type.update({'S': 'Schottky diode'})
        dev_type.update({'Ds': 'Diode with single-side barrier', 'Dd': 'Diode with double-side barrier'})
        dev_type.update({'RDs': 'Resistive diode with single-side barrier', 'RDd': 'Resistive diode with double-side barrier'})
        return dev_type

    def __init_func(self) -> dict:
        """Initialization of functions for regression"""
        func_type = {'RDs': self._func2reg_resistive_schottky_single}
        func_type.update({'RDd': self._func2reg_resistive_schottky_double})
        return func_type

    def _resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
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

    def _capacitor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
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

    def _inductor(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of an electrical inductor
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = cumtrapz(du, dx=1 / self._settings.fs_ana, initial=0) / self._settings.dev_value
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _diode_schottky_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a single-side schockly diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        is0 = 1e-12
        n0 = 8

        du = u_inp - u_inn
        i_out = is0 * np.exp(du / (n0 * self.temperature_voltage))
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _diode_single_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a diode with single-side barrier
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        is0 = 1e-12
        n0 = 8
        u_th = 0.55

        du = u_inp - u_inn
        i_out = is0 * np.exp((du - u_th) / (n0 * self.temperature_voltage))
        if self._settings.noise_en:
            i_out += self._gen_noise_awgn_pwr(du.size)
        return i_out

    def _diode_double_barrier(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
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

    def _func2reg_resistive_schottky_single(self, i_path: np.ndarray, params: list, xd: np.ndarray) -> np.ndarray:
        """Function for do least_squared regression
        Args:
            i_path: Input current sample (--> output value)
            params: Device Parameter [Is0, n, Uth, Rs]
            xd:     Input difference voltage sample
        Returns:
            Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        i_dut = np.zeros(i_path.shape) + i_path
        i_dut[i_path < 0] = 0.0

        v1 = params[2] + params[1] * self.temperature_voltage * np.log(i_dut / params[0] + 1)
        v3 = params[3] * i_path
        return xd - v1 - v3

    def _resistive_schottky_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and single-side schottky diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = [10e-12, 9, 0.0, self._settings.dev_value]

        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)
        if self._poly_fit.size == 1:
            self._get_params_polyfit(params, self._bounds_voltage, self._bounds_current,
                                     num_poly_order=11,
                                     num_points_regression=1000, do_test=True)
        return np.polyval(self._poly_fit, du)

    def _func2reg_resistive_schottky_double(self, i_path: np.ndarray, params: list, xd: np.ndarray) -> np.ndarray:
        """Function for do least_squared regression
        Args:
            i_path: Input current sample (--> output value)
            params: Device Parameter [Is0, n, Uth, Rs]
            xd:     Input difference voltage sample
        Returns:
            Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        v1 = params[2] + params[1] * self.temperature_voltage * np.log(np.abs(i_path) / params[0] + 1)
        v3 = params[3] * i_path
        return xd - np.sign(i_path) * v1 - v3

    def _resistive_schottky_double(self, u_inp: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and double-side schottky diode
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = [10e-12, 9, 0.0, self._settings.dev_value]

        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)
        if self._poly_fit.size == 1:
            self._get_params_polyfit(params, self._bounds_voltage, self._bounds_current,
                                     num_poly_order=11,
                                     num_points_regression=1000, do_test=True)
        return np.polyval(self._poly_fit, du)


# --------------------- TEST CASE ---------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    settings = SettingsDEV(
        type='RDd',
        fs_ana=1000e3,
        noise_en=False,
        para_en=False,
        dev_value=10e3,
        temp=300
    )

    # --- Declaration of input
    do_ylog = False
    t_end = 0.5e-3
    t0, uinp = _generate_signal(0.5e-3, settings.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 2.5)
    uinn = 0.0

    # --- Model declaration
    dev = ElectricalLoad(settings)
    dev.print_types()

    iout = dev.get_current(uinp, uinn)
    iin = 1e-7 * uinp
    uout = dev.get_voltage(iin, uinn, 1e-2)

    # --- Plotting: Current response
    plt.close('all')
    _plot_test_results(t0, uinp-uinn, iout, False, do_ylog)
    _plot_test_results(t0, uout+uinn, iin, True, do_ylog)
    plt.show()
