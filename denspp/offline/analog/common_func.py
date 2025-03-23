import numpy as np
from copy import deepcopy
from fxpmath import Fxp, Config


class CommonAnalogFunctions:
    _range: list = (-5.0, 5.0)

    def __init__(self) -> None:
        pass

    def define_voltage_range(self, volt_hgh: float, volt_low: float) -> list:
        """Defining the voltage range values"""
        self._range = [volt_low, volt_hgh]
        return self._range
        
    @property
    def vcm(self) -> float:
        """Returning the common mode voltage value"""
        return (self._range[0] + self._range[1]) / 2

    def clamp_voltage(self, uin: np.ndarray | float) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uout = deepcopy(uin)
        np.clip(uout, a_max=self._range[1], a_min=self._range[0], out=uout)
        return uout


class CommonDigitalFunctions:
    _digital_border: np.ndarray
    _config_fxp: Config
    _bitwidth: list = (2, 0)
    _bitsigned: bool = False

    def __init__(self) -> None:
        self._config_fxp = Config()
        self._config_fxp.rounding = "around"
        self._config_fxp.overflow = "saturate"
        self._config_fxp.underflow = "saturate"


    def define_limits(self, bit_signed: bool, total_bitwidth: int, frac_bitwidth: int) -> np.ndarray:
        """Defining the digital limitation values
        :param bit_signed:      Integer data type (unsigned: False, signed: True)
        :param total_bitwidth:  Total bitwidth
        :param frac_bitwidth:   Fraction bitwidth
        :return:                Numpy array with range (min, max)
        """
        if total_bitwidth < 0 or frac_bitwidth < 0:
            raise ValueError("total_bitwidth and frac_bitwidth must be positive")
        else:
            self._bitwidth = [total_bitwidth, frac_bitwidth]
            self._bitsigned = bit_signed

            self._digital_border = self.quantize_fxp(xin=np.array([-np.inf, np.inf]))
            return self._digital_border

    def clamp_digital(self, xin: np.ndarray) -> np.ndarray:
        """Do digital clamping of input data values
        :param xin:     Input data stream
        :return:        Output data stream
        """
        xout = deepcopy(xin)
        np.clip(xout, a_min=self._digital_border[0], a_max=self._digital_border[1], out=xout)
        return xout

    def quantize_fxp(self, xin: np.ndarray | float) -> np.ndarray:
        """Do signed quantization of input with full precision
        :param xin:     Input data stream
        :return:        Quantized output data stream
        """
        val = Fxp(val=xin, signed=self._bitsigned, n_word=self._bitwidth[0], n_frac=self._bitwidth[1],
                   config=self._config_fxp).get_val()
        return val if not type(xin) == type(float(1.2)) else float(val)
