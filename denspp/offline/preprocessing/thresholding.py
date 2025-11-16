import numpy as np
from dataclasses import dataclass
from logging import getLogger, Logger


@dataclass
class SettingsThreshold:
    """Dataclass for defining the funcs for determining properties to calculate thresholding
    Attributes:
        method:         Applied method for thresholding ['const': constant given value,
                        'abs_mean': absolute mean value, 'mad': median absolute derivation, 'mavg', moving average,
                        'mavg_abs': absolute mean absolute value, 'rms_norm': Root-Mean-Squared,
                        'rms_move': Moving RMS, 'rms_black': RMS method used in Blackrock Neurotechnology Systems,
                        'welford': Welford Online Algorithm for STD Calculation]
        sampling_rate:  Sampling rate of the transient signal [Hz]
        gain:           Applied gain on threshold output
        window_sec:     Window length in sec.
    """
    method: str
    sampling_rate:  float
    gain: float
    window_sec: float

    @property
    def window_steps(self) -> int:
        """Getting the stepsize of the window"""
        return int(self.window_sec * self.sampling_rate)


DefaultSettingsThreshold = SettingsThreshold(
    method="const",
    sampling_rate=20e3,
    gain=1.0,
    window_sec=10e-3
)


class Thresholding:
    def __init__(self, settings: SettingsThreshold) -> None:
        """Class for calculating the thresholding values based on the transient input signal
        :param settings:    Class SettingsThreshold for configuring the properties
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings: SettingsThreshold = settings
        self._methods = {
            'const':    '_constant',
            'abs_mean': '_absolute_median',
            'mad':      '_median_absolute_derivation',
            'mavg':     '_moving_average',
            'mavg_abs': '_moving_absolute_average',
            'rms_norm': '_root_mean_squared_normal',
            'rms_black':'_root_mean_squared_blackrock',
            'welford':  '_welford_online',
            'wins':     '_winsorization',
        }

    def get_overview(self) -> list:
        """Getting an overview of available thresholding methods
        :return: List with names of available methods
        """
        avai_methods = [key.lower() for key in self._methods.keys()]
        return avai_methods

    def print_overview(self) -> None:
        self._logger.info(f"Available Thresholding methods: {self.get_overview()}")

    def get_threshold(self, xin: np.ndarray, do_abs: bool=False, **kwargs) -> np.ndarray:
        """Function for getting the thresholding value from input
        :param xin:     Numpy array with transient raw signal
        :param do_abs:  Apply absolute xin for thresholding or not
        :return:        Numpy array with thresholding value from applied method
        """
        if self._settings.method.lower() == 'const' and not 'thr_val' in kwargs.keys():
            raise TypeError("Constant threshold method needs the definition of 'thr_val' (threshold value) "
                            "as float, like thr_val=0.5 in kwargs")

        if self._settings.method.lower() not in self.get_overview():
            raise ValueError(f"Thresholding method {self._settings.method} not available - Please change to {self.get_overview()}")
        xin0 = np.abs(xin) if do_abs else xin
        return getattr(self, self._methods[self._settings.method])(xin0, **kwargs)

    def get_threshold_position(self, xin: np.ndarray, pre_time: float=0.0, do_abs: bool=False, **kwargs) -> np.ndarray:
        """Function for getting the crosspoints of thresholding value and transient input
        :param xin:         Numpy array with transient raw signal
        :param pre_time:    Floating value with pre-time in the window before event is detected [s]
        :param do_abs:      Boolean for applying absolute xin for getting position and threshold
        :return:            Numpy array with thresholding value from applied method
        """
        xin0 = np.abs(xin) if do_abs else xin
        xthr = self.get_threshold(xin0, do_abs, **kwargs)
        if xthr.min() < 0:
            pos = np.argwhere(xin0 < xthr).flatten()
        else:
            pos = np.argwhere(xin0 >= xthr).flatten()
        pos_pre = int(self._settings.sampling_rate * pre_time)
        return np.array(self._get_values_non_incremented_change(pos)) - pos_pre

    @staticmethod
    def _get_values_non_incremented_change(data: np.ndarray) -> list:
        """Returns values that are not incremented by one from the previous value.
        Always includes the first element.
        """
        data0 = data.tolist()
        if not data0:
            return []
        else:
            return [data0[0]] + [data0[i] for i in range(1, len(data0)) if data0[i] != data0[i - 1] + 1]

    def _constant(self, xin: np.ndarray, thr_val: float) -> np.ndarray:
        return np.zeros_like(xin) + thr_val

    def _absolute_median(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + self._settings.gain * np.median(np.abs(xin), axis=0)

    def _median_absolute_derivation(self, xin: np.ndarray) -> np.ndarray:
            return np.zeros_like(xin) + self._settings.gain * np.median(np.abs(xin - np.mean(xin)) / 0.6745, axis=0)

    def _moving_average(self, xin: np.ndarray) -> np.ndarray:
        M = self._settings.window_steps
        conv = np.convolve(xin, np.ones(M)/M, mode='same')
        return self._settings.gain * conv

    def _moving_absolute_average(self, xin: np.ndarray) -> np.ndarray:
        M = self._settings.window_steps
        conv = np.convolve(np.abs(xin), np.ones(M)/M, mode='same')
        return self._settings.gain * conv

    def _root_mean_squared_normal(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + self._settings.gain * np.sqrt(np.sum(xin ** 2) / xin.size)

    def _root_mean_squared_blackrock(self, xin: np.ndarray) -> np.ndarray:
        return 4.5 * self._root_mean_squared_normal(xin)

    def _welford_online(self, xin: np.ndarray) -> np.ndarray:
        n = 0
        mean = 0.0
        sigma = 0.0
        std_out = np.zeros_like(xin)

        for idx, x in enumerate(xin):
            n += 1
            mean_old = mean
            mean += (x - mean) / n
            sigma += ((x - mean)* (x - mean_old) - sigma) / n
            std_out[idx] = sigma

        std_out[0:1] = std_out[2]
        return self._settings.gain * np.sqrt(std_out)
