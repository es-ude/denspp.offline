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
        min_value:      Minimal value for applying thresholding
        window_sec:     Window length in sec.
    """
    method: str
    sampling_rate:  float
    gain: float
    min_value: float | int
    window_sec: float

    @property
    def window_steps(self) -> int:
        """Getting the stepsize of the window"""
        return int(self.window_sec * self.sampling_rate)


DefaultSettingsThreshold = SettingsThreshold(
    method="const",
    sampling_rate=20e3,
    gain=1.0,
    min_value=10,
    window_sec=10e-3
)


class Thresholding:
    def __init__(self, settings: SettingsThreshold) -> None:
        """"""
        self._logger: Logger = getLogger(__name__)
        self._settings: SettingsThreshold = settings
        self._methods = {
            'const':    '_constant',
            'abs_mean': '_absolute_median',
            'mad':      '_median_absolute_derivation',
            'mavg':     '_moving_average',
            'mavg_abs': '_moving_absolute_average',
            'rms_norm': '_root_mean_squared_normal',
            'rms_move': '_root_mean_squared_moving',
            'rms_black': '_root_mean_squared_blackrock',
            'welford':  '_welford_online',
            'wins':     '_winsorization',
        }

    def get_overview(self) -> list:
        """Getting an overview of available thresholding methods
        :return: List with names of available methods
        """
        avai_methods = [key.lower() for key in self._methods.keys()]
        self._logger.info(f"Available Thresholding methods: {avai_methods}")
        return avai_methods

    def get_threshold(self, xin: np.ndarray, do_abs: bool=False) -> np.ndarray:
        """Function for getting the thresholding value from input
        :param xin:     Numpy array with transient raw signal
        :param do_abs:  Apply absolute xin for thresholding or not
        :return:        Numpy array with thresholding value from applied method
        """
        if self._settings.method.lower() not in self.get_overview():
            raise ValueError(f"Thresholding method {self._settings.method} not available - Please change to {self.get_overview()}")
        xin0 = np.abs(xin) if do_abs else xin
        return getattr(self, self._methods[self._settings.method])(xin0)

    def get_threshold_position(self, xin: np.ndarray, do_abs: bool=False) -> np.ndarray:
        """Function for getting the crosspoints of thresholding value and transient input
        :param xin:         Numpy array with transient raw signal
        :param do_abs:      Boolean for applying absolute xin for getting position and threshold
        :return:            Numpy array with thresholding value from applied method
        """
        xin0 = np.abs(xin) if do_abs else xin
        xthr = self.get_threshold(xin0, do_abs)
        if xthr.min() < 0:
            pos = np.argwhere(xin0 < xthr).flatten()
        else:
            pos = np.argwhere(xin0 >= xthr).flatten()
        return np.array(self._get_values_non_incremented_change(pos))

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

    def _constant(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + self._settings.min_value

    def _absolute_median(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + np.median(np.abs(xin), axis=0)

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

    def _root_mean_squared_moving(self, xin: np.ndarray) -> np.ndarray:
        M = self._settings.window_steps
        conv = np.convolve(xin ** 2, np.ones(M) / M, mode='same')
        return self._settings.gain * np.sqrt(conv)

    def _welford_online(self, xin: np.ndarray) -> np.ndarray:
        n = 0
        mean = 0.0
        M2 = 0.0
        std_out = np.zeros_like(xin)

        for idx, x in enumerate(xin):
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2
            std_out[idx] = M2 / (n - 1) if n > 1 else 0

        std_out[0:1] = std_out[2]
        return self._settings.gain * std_out
