import numpy as np
from logging import getLogger
from .dev_handler import ElectricalLoadHandler, SettingsDEV
from .dev_noise import ProcessNoise, SettingsNoise


DefaultSettingsDEVResistor = SettingsDEV(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    use_mode=0,
    dev_value={'r': 100e3},
    temp=300
)


DefaultSettingsDEVResistiveDiodeSingle = SettingsDEV(
    type='RDs',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    use_mode=0,
    dev_value={'i_sat': 1e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3},
    temp=300
)


DefaultSettingsDEVResistiveDiodeDouble = SettingsDEV(
    type='RDd',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    use_mode=0,
    dev_value={'i_sat': 1e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3},
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

    _fit_done: bool
    _fit_options: list
    _type_device: dict
    _bounds_curr: list
    _bounds_volt: list

    def __init__(self, settings_dev: SettingsDEV, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._init_class()
        self._logger = getLogger(__name__)

        self._settings = settings_dev
        self._fit_options = [6, 1001]

        # --- Registering electrical devices
        self.register_device(
            short_label='R',
            description='Resistor',
            param=['r'],
            func_equa=self._equation_resistor,
            func_reg=self._func2reg_resistor,
            func_fit=self._func2cur_resistor
        )
        self.register_device(
            short_label='RDs',
            description='Resistive diode (single)',
            param=['i_sat', 'n_eff', 'r_sh', 'uth0'],
            func_equa=self._equation_resistive_diode_single,
            func_reg=self._func2reg_resistive_diode,
            func_fit=self._func2cur_resistive_diode
        )
        self.register_device(
            short_label='RDd',
            description='Resistive diode (anti-parallel)',
            param=['i_sat', 'n_eff', 'r_sh', 'uth0'],
            func_equa=self._equation_resistive_diode_antiparallel,
            func_reg=self._func2reg_resistive_diode,
            func_fit=self._func2cur_resistive_diode
        )

    def _equation_resistor(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict) -> np.ndarray:
        """Performing the behaviour of an electrical resistor
        Args:
            u_inp:      Positive input voltage [V]
            u_inn:      Negative input voltage [V]
            params:     Dictionary with device parameters
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        i_out = du / params['r']
        if self._settings.noise_en:
            i_out += self.gen_noise_awgn_curr(du.size, params['r'])
        return i_out

    @staticmethod
    def _func2cur_resistor(i_device: np.ndarray, r: float) -> np.ndarray:
        """Function for fitting the electrical inputs (voltage/ current) to model parameters for a resistor
        :param i_device:    Numpy array with current parameters
        :param r:           Resistance
        :return:            Numpy array with extracted voltage
        """
        return r * i_device

    def _func2reg_resistor(self, i_path: np.ndarray, xd: np.ndarray, params: dict) -> np.ndarray:
        """Function for do least_squared regression
        Args:
            i_path:     Input current sample (--> output value)
            xd:         Input difference voltage sample
            params:     Dictionary with device parameters
        Returns:
            Numpy value with corresponding difference voltage (goes to zero with further optimization)
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
        return uth0 + r_sh * i_path + n_eff * self._settings.temperature_voltage * np.log(i_path / i_sat + 1)

    def _func2reg_resistive_diode(self, i_path: np.ndarray, xd: np.ndarray, params: dict) -> np.ndarray:
        """Function for do least_squared regression
        :param i_path:  Input current sample (--> output value)
        :param xd:      Input difference voltage sample
        :return:        Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        v1 = self._func2cur_resistive_diode(i_path, uth0=params['uth0'], n_eff=params['n_eff'],i_sat=params['i_sat'], r_sh=params['r_sh'])
        return xd - v1

    def _get_current_resistive_diode(self, u_inp: np.ndarray, u_inn: np.ndarray | float, params: dict, mode: bool) -> np.ndarray:
        """Performing the behaviour of a series connection of resistor and a diode
        Args:
            u_inp:          Positive input voltage [V]
            u_inn:          Negative input voltage [V]
            params:         Dictionary with device parameters
            mode:           Single side Connection (False) or anti-parallel Connection (True)
        Returns:
            Corresponding current signal
        """
        self._bounds_volt = [1.0, self._bounds_volt[1]]
        self._bounds_curr = [-15, -2]

        du = u_inp - u_inn if not mode else np.abs(u_inp - u_inn)
        if isinstance(du, float):
            du = np.zeros(shape=(1,))
            du[0] = u_inp - u_inn

        i_fit = self._do_regression(u_inp, u_inn)

        # --- Checking if voltage limits or current limits are reached
        xpos_uth = np.argwhere(du < 0.0).flatten()
        if not xpos_uth.size == 0:
            i_fit[xpos_uth] = params['i_sat']
        xpos_i0 = np.argwhere(i_fit < 0.0).flatten()
        if not xpos_i0.size == 0:
            i_fit[xpos_i0] = params['i_sat']

        # --- Adding noise
        if self._settings.noise_en:
            i_fit += self.gen_noise_awgn_curr(du.size, params['r_sh'])
        return i_fit
