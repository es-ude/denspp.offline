import numpy as np


class CommonAnalogFunctions:
    def __init__(self, settings) -> None:
        self._settings = settings

    def voltage_clipping(self, uin: np.ndarray | float) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uout = np.zeros(uin.shape) + uin
        if uin.size == 1:
            uout = uout if not uin > self._settings.vdd else self._settings.vdd
            uout = uout if not uin < self._settings.vss else self._settings.vss
        else:
            xpos = np.argwhere(uin > self._settings.vdd)
            xneg = np.argwhere(uin < self._settings.vss)
            uout[xpos] = self._settings.vdd
            uout[xneg] = self._settings.vss
        return uout


class CommonDigitalFunctions:
    __digital_border: np.ndarray

    def __init__(self, settings, digital_boarder: np.ndarray) -> None:
        self._settings = settings


    def digital_clipping(self, xin: np.ndarray) -> np.ndarray:
        """Do digital clipping of quantizied values"""
        xout = xin.astype('int16') if self._settings.type_out == "signed" else xin.astype('uint16')
        xout[xin > self.__digital_border[1]] = self.__digital_border[1]
        xout[xin <= self.__digital_border[0]] = self.__digital_border[0]
        return xout
