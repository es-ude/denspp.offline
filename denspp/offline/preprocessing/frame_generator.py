import numpy as np
from dataclasses import dataclass
from logging import Logger, getLogger
from fxpmath import Config, Fxp
from .thresholding import SettingsThreshold, Thresholding


@dataclass
class FrameWaveform:
    waveform: np.ndarray
    xpos: np.ndarray
    label: np.ndarray
    sampling_rate: float

    @property
    def length(self) -> int:
        return self.waveform.shape[1]

    @property
    def num_samples(self) -> int:
        return self.xpos.size

    @property
    def is_data_labeled(self) -> bool:
        return np.unique(self.label).size > 0 and 255 not in self.label.tolist()


@dataclass
class SettingsFrame:
    """Class with settings for the FrameGenerator to configure his properties
    Attributes:
        mode_thr:       String with used method for thresholding ['const': constant given value,
                        'abs_mean': absolute mean value, 'mad': median absolute derivation, 'mavg', moving average,
                        'mavg_abs': absolute mean absolute value, 'rms_norm': Root-Mean-Squared,
                        'rms_move': Moving RMS, 'rms_black': RMS method used in Blackrock Neurotechnology Systems,
                        'welford': Welford Online Algorithm for STD Calculation]
        mode_align:     Aligning mode of the detected spike frames [none, max, min,
                        ptp (Positive turning point), ntp (Negative turning point), abs-max (Absolute maximum)]
        sampling_rate:  Sampling rate of the transient signal [Hz]
        window_sec:     Time length of the frame waveform [s]
        offset_sec:     Time length for looking on the aligned position before and after the window_sec on the transient signal [s]
        align_sec:      Starting position for aligning the frame waveform [s]
        thr_gain:       Float with additional scaling value applied on the threshold value [hyperparameter]
    """
    mode_align: str
    mode_thr: str
    sampling_rate: float
    window_sec: float
    offset_sec: float
    align_sec: float
    thr_gain: float

    @property
    def length_frame_int(self) -> int:
        return int(self.window_sec * self.sampling_rate)

    @property
    def length_align_position(self) -> int:
        return int(self.align_sec * self.sampling_rate)

    @property
    def length_offset_int(self) -> int:
        return int(self.offset_sec * self.sampling_rate)

    @property
    def length_total_frame(self) -> int:
        return self.length_frame_int + 2 * self.length_offset_int


DefaultSettingsFrame = SettingsFrame(
    mode_thr='const',
    mode_align='max',
    sampling_rate=20e3,
    window_sec=2e-3,
    offset_sec=0.1e-3,
    align_sec=0.4e-3,
    thr_gain=1.0,
)


class FrameGenerator:
    def __init__(self, settings: SettingsFrame) -> None:
        """Class for generating and aligning frame woveform from a transient signal
        :param settings: Class SettingsSDA for defining the properties
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings
        self._threshold = Thresholding(settings=SettingsThreshold(
            method=self._settings.mode_thr,
            sampling_rate=self._settings.sampling_rate,
            gain=self._settings.thr_gain,
            window_sec=2*self._settings.window_sec,
        ))

    def _frame_align_none(self, frame_in: np.ndarray) -> int:
        return self._settings.length_offset_int

    def _frame_align_max(self, frame_in: np.ndarray) -> int:
        x_start = np.argmax(frame_in, axis=0)
        return x_start - self._settings.length_align_position

    def _frame_align_min(self, frame_in: np.ndarray) -> int:
        x_start = np.argmin(frame_in, axis=0)
        return x_start - self._settings.length_align_position

    def _frame_align_ptp(self, frame_in: np.ndarray) -> int:
        max_pos = 1 + np.argmax(np.diff(frame_in), axis=0)
        return max_pos - self._settings.length_align_position

    def _frame_align_ntp(self, frame_in: np.ndarray) -> int:
        max_pos = 1 + np.argmin(np.diff(frame_in), axis=0)
        return max_pos - self._settings.length_align_position

    def _frame_align_absmax(self, frame_in: np.ndarray) -> int:
        x_max = np.argmax(frame_in, axis=0)
        x_min = np.argmin(frame_in, axis=0)
        x_start = int(np.min([x_max, x_min]))
        return x_start - self._settings.length_align_position

    def get_methods_frame_aligning(self) -> list:
        """Function for getting a list with all methods for frame aligning"""
        split_key = '_frame_align_'
        return [method.split(split_key)[-1] for method in dir(self) if split_key in method]

    def get_aligning_position(self, frame_in: np.ndarray) -> int:
        """Extracting aligning position of spike frames
        :param frame_in:    Numpy array with detected spike frames
        :return:            Integer with starting position
        """
        method = f'_frame_align_{self._settings.mode_align.lower()}'
        if method in self.get_methods_frame_aligning():
            raise ValueError(f"Frame Aligning Method '{self._settings.mode_align.lower()}' is not in {self.get_methods_frame_aligning()}. Please change!")
        return getattr(self, method)(frame_in)

    # --------- Frame Generation -------------
    def get_threshold(self, xin: np.ndarray, do_abs: bool=False, **kwargs) -> np.ndarray:
        """Function for returning the threshold array in dependency of the transient input
        :param xin:     Numpy array with the transient raw input
        :param do_abs:  Boolean flag to apply absolute input for thresholding
        :return:        Numpy array with threshold value
        """
        return self._threshold.get_threshold(
            xin=xin,
            do_abs=do_abs,
            **kwargs
        )

    def get_threshold_position(self, xin: np.ndarray, do_abs: bool=False, **kwargs) -> np.ndarray:
        """Function for returning the positions of the crossing-points between input and threshold
        :param xin:     Numpy array with the transient raw input
        :param do_abs:  Boolean flag to apply absolute input for thresholding
        :return:        Numpy array with threshold value
        """
        return self._threshold.get_threshold_position(
            xin=xin,
            do_abs=do_abs,
            **kwargs
        )

    def __frame_extraction(self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int = 0) -> FrameWaveform:
        f0 = self._settings.length_offset_int
        f1 = f0 + int(self._settings.length_frame_int)

        alig_frames = list()
        alig_xpos = list()
        for idx, pos in enumerate(xpos):
            # Cutting larger frame from transient stream
            x_neg0: int = pos - self._settings.length_offset_int + xoffset
            x_pos0: int = x_neg0 + self._settings.length_total_frame
            if x_neg0 < 0 or x_pos0 > xraw.size:
                continue
            frame0 = xraw[x_neg0:x_pos0]

            # Cutting aligned frame from transient stream
            x_neg1: int = x_neg0 + f0 + self.get_aligning_position(frame0[f0:f1])
            x_pos1: int = x_neg1 + self._settings.length_frame_int
            if x_neg1 < 0 or x_pos1 > xraw.size:
                continue
            frame1 = xraw[x_neg1:x_pos1]

            alig_frames.append(frame1)
            alig_xpos.append(x_neg1)
        return FrameWaveform(
            waveform=np.array(alig_frames),
            xpos=np.array(alig_xpos),
            label=np.zeros(len(alig_xpos, ), dtype=np.dtype('uint8')) + 255,
            sampling_rate=self._settings.sampling_rate
        )

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, do_abs: bool=False, **kwargs) -> FrameWaveform:
        """Frame generation of SDA output and threshold
        :param xraw:    Numpy array with transient raw data
        :param xsda:    Numpy array with transient signal from spike detection algorithm
        :param do_abs:  Boolean for applying absolute input for thresholding
        :return:        Class FrameWaveform with waveforms, positions and labels
        """
        xthr = self._threshold.get_threshold(
            xin=xsda,
            do_abs=do_abs,
            **kwargs
        )
        xpos = self._threshold.get_threshold_position(
            xin=xsda,
            pre_time=self._settings.offset_sec,
            do_abs=do_abs,
            **kwargs
        )
        return self.__frame_extraction(
            xraw=xraw,
            xpos=xpos,
            xoffset=0
        )

    def frame_generation_with_position(self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int) -> FrameWaveform:
        """Frame generation from already detected positions (in datasets with groundtruth)
        :param xraw:    Numpy array with transient raw data
        :param xpos:    Numpy array with position where a spike frame is available
        :param xoffset: Integer value with offset to generate larger spike windows
        :return:        Tuple with [0] original (large) spike frame, [1] algined spike frame and [2] positions

        """
        return self.__frame_extraction(
            xraw=xraw,
            xpos=xpos,
            xoffset=xoffset
        )

    @staticmethod
    def do_frame_quantization(frames: np.ndarray, bit_total: int, bit_frac: int, signed: bool) -> np.ndarray:
        """Quantize the frame for sending it to hardware
        :param frames:      Numpy array with the frame waveforms [shape=(num. of waveforms, samples for each waveform)]
        :param bit_total:   Integer of the total width
        :param bit_frac:    Integer of the fraction of the total width for fixed-point number representation
        :param signed:      Boolean for signed or unsigned of the number representation
        :return:            Numpy array with the quantized frame waveform
        """
        fxp_config = Config()
        return Fxp(val=frames,
                   signed=signed,
                   n_word=bit_total,
                   n_frac=bit_frac,
                   fxp_config=fxp_config
                   ).get_val()
