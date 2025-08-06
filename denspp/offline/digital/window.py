import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
from denspp.offline.data_process.transformation import transformation_window_method


@dataclass
class SettingsWindow:
    """Class for defining the properties for applying a window sequenzer on transient signals
    Attributes:
        sampling_rate:  Floating value with sampling rate of the transient signal [Hz]
        window_sec:     Floating value with the size of the window [s]
        overlap_sec:    Floating value with overlapping the sequences [s]
    """
    sampling_rate: float
    window_sec: float
    overlap_sec: float

    @property
    def window_length(self) -> int:
        """Returning an integer with total number of samples for building the window sequence"""
        assert self.window_sec > 0, "Window length must be greater than zero"
        return int(abs(self.window_sec * self.sampling_rate))

    @property
    def overlap_length(self) -> int:
        """Returning an integer with total number of samples for overlapping"""
        assert self.overlap_sec < self.window_sec, "Overlapping size should be smaller than window size"
        return int(abs(self.overlap_sec * self.sampling_rate))


class WindowSequencer:
    _settings: SettingsWindow
    _window_normalization: np.ndarray

    def __init__(self, settings: SettingsWindow) -> None:
        """Class for applying a window sequenzer on transient signals
        :param settings:    Class SettingsWindow with definitions for the window sequenzer
        :return:            None
        """
        self._settings = settings
        self._window_normalization = transformation_window_method(
            window_size=self._settings.window_length,
            method=''
        )

    def sequence(self, signal: np.ndarray) -> np.ndarray:
        """Building a sequence-to-sequence output array from signal input"""
        num_sequences = int(signal.shape[0] / self._settings.window_length)
        array_length = num_sequences * self._settings.window_length
        return signal[0:array_length].reshape(
            (num_sequences, self._settings.window_length),
            copy=True
        )

    def slide(self, signal: np.ndarray) -> np.ndarray:
        """Building a sliding window sequencer on signal input"""
        delta_steps = self._settings.window_length - self._settings.overlap_length
        return sliding_window_view(
            x=signal,
            axis=0,
            window_shape=self._settings.window_length,
            writeable=True
        )[::delta_steps]
