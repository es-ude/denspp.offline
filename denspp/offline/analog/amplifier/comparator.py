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
        offset:     Offset voltage of the amplifier [V] without VCM,
        noise_dis:  Voltage Noise distribution of comparator [V/sqrt(Hz)]
        hysteresis: Hysteresis voltage window of comparator [%]
        out_analog: Is output analog [True] or digital [False]
        out_invert: Is output active low [True] or active high [False]
    """
    vdd:    float
    vss:    float
    # Comparator characteristics
    gain: int
    offset: float
    noise_dis: float
    hysteresis: float
    out_analog: bool
    out_invert: bool


DefaultSettingsComparator = SettingsComparator(
    vdd=0.6, vss=-0.6,
    gain=100,
    offset=-1e-3,
    noise_dis=0.1e-3,
    hysteresis=0.25,
    out_analog=False,
    out_invert=False,
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

    def __generate_noise(self, input: int, scale: float=0.1, use_noise: bool=True) -> np.ndarray:
        return np.random.normal(self._settings.offset, scale, input) if use_noise else self._settings.offset * np.ones(input)

    def __apply_gain_comparator(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        du = np.array(uinp - uinn)
        return self.vcm + self._settings.gain * (du - self._settings.offset)

    def __generate_output(self, ucmp: np.ndarray) -> np.ndarray:
        return np.array(ucmp >= self.vcm + self._settings.offset) if not self._settings.out_analog else ucmp

    def __apply_inverter(self, u_cmp: np.ndarray | float) -> np.ndarray:
        return (np.invert(u_cmp) if not self._settings.out_analog else 2* self.vcm - u_cmp) if self._settings.out_invert else u_cmp

    def __apply_hysteresis(self, du: np.ndarray, mode: int) -> np.ndarray:
        """Processing differential input for generating hysteresis"""
        thr = self.__type_hysteresis(mode)

        u_out = np.zeros(du.shape)
        u_out += self._unoise
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        # state == 0 --> not active, state == 1 --> active
        for idx, val in enumerate(du):
            thr_run = thr[0] if not self._int_state[idx] else thr[1]
            out = np.array(val - thr_run)
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1
            u_out[idx] = self.vcm + self._settings.gain * out
        return u_out

    def __type_hysteresis(self, mode: int) -> list:
        """Definition of type"""
        thr_zero = self._settings.offset
        thr_pos = thr_zero + self._settings.hysteresis * (self._settings.vdd - self.vcm)
        thr_neg = thr_zero + self._settings.hysteresis * (self._settings.vss - self.vcm)

        self.__logger.debug(f"Pos. hysterese window voltage at: {thr_pos} V")
        self.__logger.debug(f"Neg. hysterese window voltage at: {thr_neg} V")
        match mode:
            case 1:
                # --- Single Side, negative VSS
                list_out = [thr_zero, thr_neg]
            case 2:
                # --- Single Side, positive VDD
                list_out = [thr_pos, thr_zero]
            case 3:
                # --- Double Side, VSS-VDD
                list_out = [thr_pos, thr_neg]
            case _:
                # --- Normal comparator
                list_out = [thr_zero, thr_zero]
        return list_out

    def cmp_ideal(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs an ideal comparator with input signal (with offset, but no noise)
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean or voltage values (depends on out_analog)
        """
        u_cmp = self.__apply_gain_comparator(uinp, uinn)
        self._unoise = self.__generate_noise(input=u_cmp.size, scale=self._settings.noise_dis, use_noise=False)
        u_cmp = self.clamp_voltage(u_cmp)
        return self.__apply_inverter(self.__generate_output(u_cmp))

    def cmp_normal(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a normal comparator with input signal (with noise and offset)
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean or voltage values (depends on out_analog)
        """
        u_cmp = self.__apply_gain_comparator(uinp, uinn)
        self._unoise = self.__generate_noise(input=u_cmp.size, scale=self._settings.noise_dis, use_noise=True)
        u_out = self.clamp_voltage(u_cmp + self._unoise)
        return self.__apply_inverter(self.__generate_output(u_out))

    def cmp_single_pos_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__generate_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__apply_hysteresis(du, 2))
        return self.__apply_inverter(self.__generate_output(u_out))

    def cmp_single_neg_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__generate_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__apply_hysteresis(du, 1))
        return self.__apply_inverter(self.__generate_output(u_out))

    def cmp_double_hysteresis(self, uinp: np.ndarray | float, uinn: np.ndarray | float) -> np.ndarray:
        """Performs a double-side hysteresis comparator with input signal (with noise and offset)
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = np.array(uinp - uinn)
        self._unoise = self.__generate_noise(du.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(self.__apply_hysteresis(du, 3))
        return self.__apply_inverter(self.__generate_output(u_out))
