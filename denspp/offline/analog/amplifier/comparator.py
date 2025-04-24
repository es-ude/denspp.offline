import dataclasses
import numpy as np
from logging import getLogger
from denspp.offline.analog.common_func import CommonAnalogFunctions


@dataclasses.dataclass
class SettingsComparator:
    """Individual data class to configure an analogue voltage comparator
    Params:
        vdd:        Positive supply voltage [V],
        vss:        Negative supply voltage [V],
        gain:       Amplification factor of comparator [V/V]
        out_analog: Is output analog [True] or digital [False]
        offset:     Offset voltage of the amplifier [V] without VCM,
        noise_dis:  Voltage Noise distribution of comparator [V/sqrt(Hz)]
        hysteresis:  Hysteresis voltage window of comparator [%]
    """
    vdd:    float
    vss:    float
    # Comparator characteristics
    gain: int
    out_analog: bool
    offset: float
    noise_dis: float
    hysteresis: float


DefaultSettingsComparator = SettingsComparator(
    vdd=0.6, vss=-0.6,
    out_analog=False,
    gain=100,
    offset=-1e-3,
    noise_dis=0.1e-3,
    hysteresis=0.25
)


class Comparator(CommonAnalogFunctions):
    _settings: SettingsComparator
    _unoise: np.ndarray
    _int_state: np.ndarray

    @property
    def get_noise_signal(self) -> np.ndarray:
        return self._unoise

    def __init__(self, settings_dev: SettingsComparator) -> None:
        """Class for emulating an analogue comparator
        :param settings_dev:    Dataclass for handling the comparator amplifier
        """
        super().__init__()
        self.__logger = getLogger(__name__)
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._settings = settings_dev

    def __gen_noise(self, input: int, scale: float=0.1, use_noise: bool=True) -> np.ndarray:
        """Generate the input noise"""
        return np.random.normal(self._settings.offset, scale, input) if use_noise else self._settings.offset * np.ones(input)

    def __cmp_calc(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performing the amplification of comparator"""
        du = np.array(uinp - uinn)
        return self.vcm + self._settings.offset + self._settings.gain * du

    def __cmp_generate_output(self, ucmp: np.ndarray) -> np.ndarray:
        """Generating the comparator output stream"""
        return np.array(ucmp >= self.vcm + self._settings.offset) if not self._settings.out_analog else ucmp

    def __cmp_hysteresis(self, du: np.ndarray, mode: int) -> np.ndarray:
        """Processing differential input for generating hysteresis"""
        thr = self.__type_hysteresis(mode)

        u_out = np.zeros(du.shape)
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        # state == 0 --> not active, state == 1 --> active
        for idx, val in enumerate(du):
            out = val - (thr[0] if not self._int_state[idx] else thr[1]) + self._unoise[idx]
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1
            u_out[idx] = self.vcm + self._settings.gain * out
        return u_out

    def __type_hysteresis(self, mode: int) -> list:
        """Definition of type"""
        thr_pos = self._settings.hysteresis * (self._settings.vdd - self.vcm)
        thr_neg = self._settings.hysteresis * (self._settings.vss - self.vcm)
        self.__logger.debug(f"Pos. hysterese window voltage at: {thr_pos} V")
        self.__logger.debug(f"Neg. hysterese window voltage at: {thr_neg} V")
        match mode:
            case 0:
                # --- Normal comparator
                list_out = [0.0, 0.0]
            case 1:
                # --- Single Side, negative VSS
                list_out = [0.0, thr_neg]
            case 2:
                # --- Single Side, positive VDD
                list_out = [thr_pos, 0.0]
            case 3:
                # --- Double Side, VSS-VDD
                list_out = [thr_pos, thr_neg]
            case _:
                list_out = [0.0, 0.0]
        return list_out

    def cmp_ideal(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs an ideal comparator with input signal (with offset, but no noise)
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean or voltage values (depends on out_analog)
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        self._unoise = self.__gen_noise(input=u_cmp.size, scale=self._settings.noise_dis, use_noise=False)
        u_cmp = self.clamp_voltage(u_cmp)
        return self.__cmp_generate_output(u_cmp)

    def cmp_normal(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a normal comparator with input signal (with noise and offset)
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean or voltage values (depends on out_analog)
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        self._unoise = self.__gen_noise(input=u_cmp.size, scale=self._settings.noise_dis, use_noise=True)
        u_out = self.clamp_voltage(u_cmp - self._unoise)
        return self.__cmp_generate_output(u_out)

    def cmp_single_pos_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, 2))
        return self.__cmp_generate_output(u_out)

    def cmp_single_neg_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, 1))
        return self.__cmp_generate_output(u_out)

    def cmp_double_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a double-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, 3))
        return self.__cmp_generate_output(u_out)
