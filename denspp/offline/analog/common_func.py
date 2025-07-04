import numpy as np
from copy import deepcopy
from fxpmath import Fxp, Config


class CommonAnalogFunctions:
    _range: list = (-5.0, 5.0)

    def define_voltage_range(self, volt_hgh: float, volt_low: float) -> list:
        """Defining the voltage range values"""
        self._range = [volt_low, volt_hgh]
        return self._range
        
    @property
    def vcm(self) -> float:
        """Returning the common mode voltage value"""
        return (self._range[0] + self._range[1]) / 2

    def clamp_voltage(self, uin: np.ndarray | float) -> np.ndarray | float:
        """Do voltage clipping at voltage supply"""
        uout = np.array(deepcopy(uin))
        np.clip(uout, a_max=self._range[1], a_min=self._range[0], out=uout)
        return float(uout) if isinstance(uin, float) else uout


class CommonDigitalFunctions:
    _digital_border: np.ndarray
    _bitwidth: list = (2, 0)
    _bitsigned: bool = False

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

    @staticmethod
    def checking_binary_limits_violation(data: np.ndarray, bitwidth: int, do_signed: bool) -> np.ndarray:
        """Function for checking if data has some binary limit violations and correct them
        :param data:        Numpy array with data for checking
        :param bitwidth:    Used bitwidth for checking
        :param do_signed:   Output is signed (True) or unsigned (False)
        :return:            Numpy array with corrected binary data
        """
        data_new = deepcopy(data)
        chck_lim = [-(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1) - 1)] if do_signed else [0, ((2 ** bitwidth) - 1)]
        if data_new.max() > chck_lim[1]:
            xpos = np.argmax(data)
            data_new[xpos] = chck_lim[1]
        if data_new.min() < chck_lim[0]:
            xpos = np.argmin(data)
            data_new[xpos] = chck_lim[0]
        return data_new

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
        config_fxp = Config()
        config_fxp.rounding = "around"
        config_fxp.overflow = "saturate"
        config_fxp.underflow = "saturate"

        val = Fxp(val=xin, signed=self._bitsigned, n_word=self._bitwidth[0], n_frac=self._bitwidth[1],
                   config=config_fxp).get_val()
        return val if not type(xin) == type(float(1.2)) else float(val)

    @staticmethod
    def extract_rising_edge(trigger: np.ndarray) -> list:
        """Extracting the rising edges of an boolean array (e.g. output signal of a comparator)
        :param trigger:     Numpy array with trigger signal (transient)
        :return:            List with index of rising edges
        """
        trgg_evnt = np.flatnonzero((trigger[:-1] == False) & (trigger[1:] == True)) + 1
        return trgg_evnt.tolist()

    @staticmethod
    def extract_falling_edge(trigger: np.ndarray) -> list:
        """Extracting the falling edges of an boolean array (e.g. output signal of a comparator)
        :param trigger:     Numpy array with trigger signal (transient)
        :return:            List with index of rising edges
        """
        trgg_evnt = np.flatnonzero((trigger[:-1] == True) & (trigger[1:] == False)) + 1
        return trgg_evnt.tolist()