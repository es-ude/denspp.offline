import dataclasses
import numpy as np
from denspp.offline.analog.common_func import CommonAnalogFunctions


@dataclasses.dataclass
class SettingsCMP:
    """Individual data class to configure an analogue comparator

    Params:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        offset:     Offset voltage of the amplifier [V]
        noise:      Enable noise on output [True/False]
    """
    vdd:    float
    vss:    float
    # Comparator characteristics
    gain: int
    out_analog: bool
    offset: float
    noise:  bool
    noise_dis: float

    @property
    def vcm(self) -> np.ndarray:
        return np.array(self.vdd + self.vss) / 2


RecommendedSettingsCMP = SettingsCMP(
    vdd=0.6, vss=-0.6,
    out_analog=False,
    gain=100,
    offset=-1e-3,
    noise=True
)


class Comp(CommonAnalogFunctions):
    _settings: SettingsCMP
    _unoise: np.ndarray
    _int_state: np.ndarray

    def __init__(self, settings_dev: SettingsCMP) -> None:
        """Class for emulating an analogue comparator
        :param settings_dev:    Dataclass for handling the comparator amplifier
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vss, volt_hgh=settings_dev.vdd)
        self._settings = settings_dev

    def __gen_noise(self, input: int, scale: float=0.1) -> np.ndarray:
        """Generate the input noise"""
        return np.random.normal(self._settings.offset, scale, input) if self._settings.noise else self._settings.offset * np.ones(input)

    def __cmp_calc(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performing the difference calculation"""
        du = uinp - uinn
        return self._settings.vcm + self._settings.gain * du

    @staticmethod
    def __dig_output(du: np.ndarray) -> np.ndarray:
        """Translating the analogue signal into digital trigger"""
        return np.array(np.sign(du) == True)

    def __cmp_hysteresis(self, du: np.ndarray, thr: list) -> np.ndarray:
        """Processing differential input for generating hysteresis"""
        u_out = np.zeros(du.shape)
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        # state == 0 --> not active, state == 1 --> active
        for idx, val in enumerate(du):
            out = val - (thr[0] if not self._int_state[idx] else thr[1]) + self._unoise[idx]
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1

            u_out[idx] = self._settings.vcm + self._settings.gain * out

        return u_out

    def __type_hysteresis(self, mode: int, scale_thr: float) -> list:
        """Definition of type"""
        list_out = [0.0, 0.0]
        match mode:
            case 0:
                # --- Normal comparator
                list_out = [0.0, 0.0]
            case 1:
                # --- Single Side, negative VSS
                list_out = [0.0, scale_thr * (self._settings.vss - self._settings.vcm)]
            case 2:
                # --- Single Side, positive VDD
                list_out = [scale_thr * (self._settings.vdd - self._settings.vcm), 0.0]
            case 3:
                # --- Double Side, VSS-VDD
                list_out = [scale_thr * (self._settings.vdd - self._settings.vcm), scale_thr * (self._settings.vss - self._settings.vcm)]
            case 4:
                # --- Double Side, Reference
                list_out = [0.0, 0.0]

        return list_out

    def cmp_ideal(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs an ideal comparator with input signal
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        return self.clamp_voltage(u_cmp)

    def cmp_normal(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs a normal comparator with input signal
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        Returns:
            Corresponding numpy array with boolean values
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        self._unoise = self.__gen_noise(u_cmp.size, self._settings.noise_dis)
        u_out = self.clamp_voltage(u_cmp - self._unoise)
        if not self._settings.out_analog:
            u_out = self.__dig_output(u_out)
        return u_out

    def cmp_single_pos_hysteresis(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr: float=0.25) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = uinp - uinn
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        thr0 = self.__type_hysteresis(2, scale_thr)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, thr0))
        if not self._settings.out_analog:
            u_out = self.__dig_output(u_out)
        return u_out

    def cmp_single_neg_hysteresis(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr: float=0.25) -> np.ndarray:
        """Performs a single-side hysteresis comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = uinp - uinn
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        thr0 = self.__type_hysteresis(1, scale_thr)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, thr0))
        if not self._settings.out_analog:
            u_out = self.__dig_output(u_out)
        return u_out

    def cmp_double_hysteresis(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr: float=0.25) -> np.ndarray:
        """Performs a double-side hysteresis comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        Returns:
            Corresponding numpy array with boolean values
        """
        du = uinp - uinn
        self._unoise = self.__gen_noise(du.size, self._settings.noise_dis)
        thr0 = self.__type_hysteresis(3, scale_thr)
        u_out = self.clamp_voltage(self.__cmp_hysteresis(du, thr0))
        if not self._settings.out_analog:
            u_out = self.__dig_output(u_out)
        return u_out
