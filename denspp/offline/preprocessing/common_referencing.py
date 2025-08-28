import numpy as np
from logging import getLogger, Logger
from dataclasses import dataclass
from scipy.signal import convolve2d


@dataclass
class SettingsReferencing:
    """Class for defining the properties of the common referencing methods
    Attributes:
        dim:            Integer with applied dimension
        kernel_size:    Kernel size for convolution (must be odd-numbered)
    """
    kernel_size: int


DefaultSettingsReferencing = SettingsReferencing(
    kernel_size=3,
)


class CommonReferencing:
    _logger: Logger

    def __init__(self, settings: SettingsReferencing) -> None:
        self._logger = getLogger(__name__)
        self._settings = settings

    def build_dummy_active_mapping(self, signal: np.ndarray) -> np.ndarray:
        """Function for building a dummy active mapper with True values
        :param signal:  Numpy array with channel-specific signals
        :return:        Numpy array with boolean (default: true) for channels are used
        """
        match len(signal.shape):
            case 1:
                return np.ones(shape=(1, ), dtype=bool)
            case 2:
                return np.ones(shape=(signal.shape[0], ), dtype=bool)
            case 3:
                return np.ones(shape=(signal.shape[0], signal.shape[1]), dtype=bool)
            case _:
                raise NotImplementedError

    def get_reference_map(self, signal: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Building the common reference mapper using CAR algorithm (Common Average Referencing) on input signals
        :param signal:  Input signal of transient analysis with shape [num_channels, num_smaples] or num_channels splitted into electrode design
        :param active:  Overview of active channels used for referencing
        :return:        Numpy array with convolved signal for doing common average referencing
        """
        match len(signal.shape):
            case 1:
                return self._calculate_reference_car_1d(signal, active)
            case 2:
                return self._calculate_reference_car_1d(signal, active)
            case 3:
                return self._calculate_reference_car_2d(signal, active)
            case _:
                raise NotImplementedError

    def apply_reference(self, signal: np.ndarray, active: np.ndarray) -> np.ndarray:
        """"""
        return signal - self.get_reference_map(signal, active)

    @staticmethod
    def _calculate_reference_car_1d(mea_signal: np.ndarray, mapp_used: np.ndarray) -> np.ndarray:
        if len(mea_signal.shape) >= 2:
            mapping = np.repeat(mapp_used[:, np.newaxis], mea_signal.shape[-1], axis=1)
        else:
            mapping = np.array([mapp_used] * mea_signal.size)
        return np.mean(mea_signal * mapping, axis=0)


    def _calculate_reference_car_2d(self, mea_signal: np.ndarray, mapp_used: np.ndarray) -> np.ndarray:
        if not len(mea_signal.shape) == 3:
            raise NotImplementedError("The input numpy array has wrong size - Please check!")
        if self._settings.kernel_size == 1:
            raise ValueError("Kernel size must greater then 1")
        if self._settings.kernel_size % 2 == 0:
            raise ValueError("Value for building the kernel in CAR algorithm must be odd-numbered!")

        # --- Generating the kernel
        kernel = np.ones((self._settings.kernel_size, self._settings.kernel_size), dtype=float)
        mid_number = int(np.floor(self._settings.kernel_size / 2))
        kernel[mid_number, mid_number] = 0.0

        kernel = kernel / np.sum(kernel)

        # --- Do the convolution
        data_out = np.zeros(mea_signal.shape, dtype=float)
        for idx in range(0, mea_signal.shape[-1]):
            data_in = mea_signal[:, :, idx]
            conv_out = convolve2d(data_in, kernel, mode='same')
            data_out[:, :, idx] = conv_out

        # --- Correction of not available channels
        for row in range(0, mea_signal.shape[0]):
            for col in range(0, mea_signal.shape[1]):
                if not mapp_used[row, col]:
                    data_out[row, col, :] = np.zeros((mea_signal.shape[-1], ), dtype=float)

        return data_out
