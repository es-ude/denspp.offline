import numpy as np
from .dev_noise import SettingsNoise
from .dev_handler import ElectricalLoadHandler, SettingsDevice



DefaultSettingsDEVResistor = SettingsDevice(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    dev_value={'r': 100e3},
    temp=300,
    use_poly=False
)


DefaultSettingsDEVResistiveDiodeSingle = SettingsDevice(
    type='RDs',
    fs_ana=50e3,
    noise_en=False,
    dev_value={'i_sat': 10e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3},
    temp=300,
    use_poly=False
)


DefaultSettingsDEVResistiveDiodeDouble = SettingsDevice(
    type='RDd',
    fs_ana=50e3,
    noise_en=False,
    dev_value={'i_sat': 1e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3},
    temp=300,
    use_poly=False
)


RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
)

class ElectricalLoad(ElectricalLoadHandler):

    def __init__(self, settings_dev: SettingsDevice, settings_noise=RecommendedSettingsNoise):
        ElectricalLoadHandler.__init__(self, settings_dev)

        # --- Registering electrical devices
        self.register_device(
            short_label='R',
            description='Resistor',
            func_equa=self._equation_resistor,
            func_reg=self._func2reg_resistor,
            func_fit=self._func2cur_resistor
        )
        self.register_device(
            short_label='RDs',
            description='Resistive diode (single)',
            func_equa=self._equation_resistive_diode_single,
            func_reg=self._func2reg_resistive_diode,
            func_fit=self._func2cur_resistive_diode
        )
        self.register_device(
            short_label='RDd',
            description='Resistive diode (anti-parallel)',
            func_equa=self._equation_resistive_diode_antiparallel,
            func_reg=self._func2reg_resistive_diode,
            func_fit=self._func2cur_resistive_diode
        )

    def _equation_resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict) -> np.ndarray:
        """Performing the behaviour of an electrical resistor
        :param u_inp:       Positive input voltage [V]
        :param u_inn:       Negative input voltage [V]
        :param params:      Dictionary with device parameters
        :return:            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = du / params['r']
        if self._settings_device.noise_en:
            i_out += self.gen_noise_awgn_curr(du.size, params['r'])
        return i_out

    def _func2cur_resistor(self, i_device: np.ndarray, r: float) -> np.ndarray:
        """Function for fitting the electrical inputs (voltage/ current) to model parameters for a resistor
        :param i_device:    Numpy array with current parameters
        :param r:           Resistance
        :return:            Numpy array with extracted voltage
        """
        return r * i_device

    def _func2reg_resistor(self, i_path: np.ndarray, xd: np.ndarray, params: dict) -> np.ndarray:
        """Function for do least_squared regression
        :param i_path:      Input current sample (--> output value)
        :param xd:          Input difference voltage sample
        :param params:      Dictionary with device parameters
        :return:            Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        v1 = self._func2cur_resistor(i_path, **params)
        return xd - v1

    def _equation_resistive_diode_single(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and a diode
        :param u_inp:       Positive input voltage [V]
        :param u_inn:       Negative input voltage [V]
        :param params:      Dictionary with device parameters
        :return:            Corresponding current signal
        """
        i_fit = self._get_current_resistive_diode(u_inp, u_inn, params, False)
        xpos_neg = np.argwhere(u_inp-u_inn < 0.0).flatten()
        if not xpos_neg.size == 0:
            i_fit[xpos_neg] = 0.0
        return i_fit

    def _equation_resistive_diode_antiparallel(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and a diode
        :param u_inp:       Positive input voltage [V]
        :param u_inn:       Negative input voltage [V]
        :param params:      Dictionary with device parameters
        :return:            Corresponding current signal
        """
        i_fit = self._get_current_resistive_diode(u_inp, u_inn, params, True)
        xpos_neg = np.argwhere(u_inp-u_inn < 0.0).flatten()
        if not xpos_neg.size == 0:
            i_fit[xpos_neg] = (-1) * i_fit[xpos_neg]

        return i_fit

    def _func2cur_resistive_diode(self, i_path: np.ndarray, uth0: float, r_sh: float, n_eff: float, i_sat: float) -> np.ndarray:
        """Function for performing curve fitting for resistive diode behaviour
        :param i_path:      Numpy array with current signal [A]
        :param uth0:        Threshold voltage of device
        :param r_sh:        Shunt resistance
        :param n_eff:       Effective emission coefficient
        :param i_sat:       Saturation current
        :return:            Corresponding voltage signal
        """
        return uth0 + r_sh * i_path + n_eff * self._settings_device.temperature_voltage * np.log(i_path / i_sat + 1)

    def _func2reg_resistive_diode(self, i_path: np.ndarray, xd: np.ndarray, params: dict) -> np.ndarray:
        """Function for do least_squared regression
        :param i_path:  Input current sample (--> output value)
        :param xd:      Input difference voltage sample
        :return:        Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        v1 = self._func2cur_resistive_diode(i_path, **params)
        return xd - v1

    def _get_current_resistive_diode(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict, mode: bool) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and a diode
        :param u_inp:   Positive input voltage [V]
        :parma u_inn:   Negative input voltage [V]
        :param params:  Dictionary with device parameters
        :param mode:    Single side Connection (False) or antiparallel Connection (True)
        :return:        Corresponding current signal
        """
        du = u_inp - u_inn if not mode else np.abs(u_inp - u_inn)
        if isinstance(du, float):
            du = np.zeros(shape=(1,))
            du[0] = u_inp - u_inn

        i_fit = self._do_regression(u_inp=u_inp, u_inn=u_inn, params=params, disable_print=True)

        # --- Checking if voltage limits or current limits are reached
        xpos_uth = np.argwhere(du < 0.0).flatten()
        if not xpos_uth.size == 0:
            i_fit[xpos_uth] = params['i_sat']
        xpos_i0 = np.argwhere(i_fit < 0.0).flatten()
        if not xpos_i0.size == 0:
            i_fit[xpos_i0] = params['i_sat']

        # --- Adding noise
        if self._settings_device.noise_en:
            i_fit += self.gen_noise_awgn_curr(du.size, params['r_sh'])
        return i_fit
