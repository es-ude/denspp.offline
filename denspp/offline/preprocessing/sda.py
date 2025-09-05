from logging import getLogger, Logger
import numpy as np
from dataclasses import dataclass
from scipy.signal import iirfilter, lfilter
from .frame_generator import FrameWaveform, SettingsFrame, FrameGenerator


@dataclass
class SettingsSDA:
    """Configuration class for defining the Spike Detection Algorithm (SDA)
    Attributes:
        mode_sda:       Applied spike detection algorithm (SDA) on transient signal [normal, Non-Linear Energy Operator (NEO) or Teager-Kaiser-Operator (dx_sda = 1 or kNEO with dx_sda > 1),
                        Multiresolution Teager Energy Operator (MTEO), absolute difference operator (ADO),
                        enhanced energy-derivation operator (eED),
                        amplitude slope operator (ASO, k for window size, and f_hp as additional float arg),
                        spike band-power estimation [Nason et al., 2020] (SBP, using f_bp with two values as additional arg)
        mode_thr:       String with used method for thresholding ['const': constant given value,
                        'abs_mean': absolute mean value, 'mad': median absolute derivation, 'mavg', moving average,
                        'mavg_abs': absolute mean absolute value, 'rms_norm': Root-Mean-Squared,
                        'rms_move': Moving RMS, 'rms_black': RMS method used in Blackrock Neurotechnology Systems,
                        'welford': Welford Online Algorithm for STD Calculation]
        mode_align:     Aligning mode of the detected spike frames [none, max, min,
                        ptp (Positive turning point), ntp (Negative turning point), abs-max (Absolute maximum)]
        sampling_rate:  Sampling rate [Hz]
        dx_sda:         Position difference for extracting SDA method. Configuration with length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        t_frame_length: Floating value with total window length [s]
        t_frame_start:  Floating value with time point for aligned position [s]
        dt_offset:      Time offset for the first larger spike window [neg, pos]
        thr_gain:       Floating value with amplification factor on SDA output
    """
    mode_sda: str
    mode_thr: str
    mode_align: str
    dx_sda: list
    sampling_rate: float
    t_frame_length: float
    t_frame_start: float
    dt_offset: float
    thr_gain: float

    @property
    def get_integer_offset(self) -> int:
        """Getting the integer offset for negative offset in building the spike window"""
        return round(self.dt_offset * self.sampling_rate)

    @property
    def get_integer_offset_total(self) -> int:
        """Getting the total integer offset in building the spike window"""
        return 2* self.get_integer_offset

    @property
    def get_integer_spike_frame(self) -> int:
        """Getting the integer for total length of a spike window"""
        return round(self.t_frame_length * self.sampling_rate)

    @property
    def get_integer_spike_start(self) -> int:
        """Getting the integer for starting the aligned method on each spike window"""
        return round(self.t_frame_start * self.sampling_rate)


DefaultSettingsSDA = SettingsSDA(
    sampling_rate=20e3,
    dx_sda=[1],
    mode_sda='eed',
    mode_thr='const',
    mode_align='min',
    t_frame_length=1.6e-3,
    t_frame_start=0.4e-3,
    dt_offset=0.1e-3,
    thr_gain=1.0
)


class SpikeDetection:
    def __init__(self, settings: SettingsSDA) -> None:
        """Class SpikeDetection for extracting Spike Waveforms from neural transient input
        :param settings:    Class SettingsSDA for configuring the accelerator
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings_sda = settings
        self._settings_thr = SettingsFrame(
            mode_thr=self._settings_sda.mode_thr,
            mode_align=self._settings_sda.mode_align,
            sampling_rate=self._settings_sda.sampling_rate,
            window_sec=self._settings_sda.t_frame_length,
            offset_sec=self._settings_sda.dt_offset,
            align_sec=self._settings_sda.t_frame_start,
            thr_gain=self._settings_sda.thr_gain
        )
        self._frame_generator = FrameGenerator(
            settings=self._settings_thr,
        )

    @staticmethod
    def _sda_normal(xin: np.ndarray) -> np.ndarray:
        return xin

    def _sda_neo(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings_sda.dx_sda[0]
        x_neo0 = xin[ksda0:-ksda0] ** 2 - xin[:-2 * ksda0] * xin[2 * ksda0:]
        return np.concatenate([x_neo0[:ksda0, ], x_neo0, x_neo0[-ksda0:, ]], axis=None)

    def _sda_mteo(self, xin: np.ndarray) -> np.ndarray:
        x_mteo = np.zeros(shape=(len(self._settings_sda.dx_sda), xin.size))
        for idx, ksda0 in enumerate(self._settings_sda.dx_sda):
            x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
            x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)
        return np.max(x_mteo, axis=0)

    def _sda_ado(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings_sda.dx_sda[0]
        x_sda = np.absolute(xin[ksda0:, ] - xin[:-ksda0, ])
        return np.concatenate([x_sda[:ksda0], x_sda], axis=None)

    def _sda_aso(self, xin: np.ndarray) -> np.ndarray:
        ksda0 = self._settings_sda.dx_sda[0]
        x_sda = xin[ksda0:, ] * (xin[ksda0:, ] - xin[:-ksda0, ])
        return np.concatenate([x_sda[:ksda0], x_sda], axis=None)

    def _sda_eed(self, xin: np.ndarray, f_hp: float) -> np.ndarray:
        filter = iirfilter(
            N=2, Wn=2 * f_hp / self._settings_sda.sampling_rate, ftype="butter", btype="highpass",
            analog=True, output='ba'
        )
        return np.square(np.array(lfilter(filter[0], filter[1], xin)))

    def _sda_spb(self, xin: np.ndarray, f_bp: list) -> np.ndarray:
        filter = iirfilter(N=2, Wn=2 * np.array(f_bp) / self._settings_sda.sampling_rate, ftype="butter", btype="bandpass", analog=False, output='ba')
        filt0 = lfilter(filter[0], filter[1], xin)
        return np.abs(filt0)

    def get_methods_sda(self) -> list:
        """Function for getting a list with all methods for spike detection"""
        split_key = '_sda_'
        return [method.split(split_key)[-1] for method in dir(self) if split_key in method]

    def apply_spike_detection(self, xraw: np.ndarray, **kwargs) -> np.ndarray:
        """Applying spike detection algorithm (SDA) on transient raw signal
        :param xraw:    Numpy array with transient raw data
        :return:        Numpy array with transient threshold value for extracting spike waveforms
        """
        if len(self._settings_sda.dx_sda) < 1:
            raise ValueError('Length of dx_sda must be greater than 1')
        if self._settings_sda.dx_sda[0] < 1:
            raise ValueError('Value of dx_sda[0] must be greater than 1')
        if self._settings_sda.mode_sda == 'eed' and not 'f_hp' in kwargs.keys():
            raise TypeError("EED method needs the definition of 'f_hp' (high-pass corner "
                            "frequency as float, like f_hp=150.) as kwargs")
        if self._settings_sda.mode_sda == 'spb' and not 'f_bp' in kwargs.keys():
            raise TypeError("SPB method needs the definition of 'f_bp' (band-pass corner "
                            "frequencies as tuple/list[float, float], like f_bp=[100., 1000.]) as kwargs")

        method = f'_sda_{self._settings_sda.mode_sda.lower()}'
        if method in self.get_methods_sda():
            raise ValueError(f"Spike Detection Method '{self._settings_sda.mode_align.lower()}' is not in {self.get_methods_sda()}. Please change!")
        return getattr(self, method)(xraw, **kwargs)

    def get_spike_waveforms(self, xraw: np.ndarray, do_abs: bool, **kwargs) -> FrameWaveform:
        """Function for extracting the spike waveforms from transient input
        :param xraw:    Numpy array with transient input
        :param do_abs:  Boolean for absolute threshold estimation
        :return:        Class FrameWaveform with waveforms, labels and position
        """
        key_kwargs = [key for key in kwargs.keys()]
        if 'f_hp' in key_kwargs and self._settings_sda.mode_sda == 'eed':
            kwargs_sda = {'f_hp': kwargs['f_hp']}
        elif 'f_bp' in key_kwargs and self._settings_sda.mode_sda == 'spb':
            kwargs_sda = {'f_bp': kwargs['f_bp']}
        else:
            kwargs_sda = dict()

        if 'thr_val' in key_kwargs and self._settings_sda.mode_thr == 'const':
            kwargs_thr = {'thr_val': kwargs['thr_val']}
        else:
            kwargs_thr = dict()

        xsda = self.apply_spike_detection(xraw=xraw, **kwargs_sda)
        return self._frame_generator.frame_generation(
            xraw=xraw,
            xsda=xsda,
            do_abs=do_abs,
            **kwargs_thr
        )

    def get_spike_waveforms_from_positions(self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int) -> FrameWaveform:
        """Function for extracting the spike waveforms from transient input and given position
        :param xraw:    Numpy array with transient input
        :param xpos:    Numpy array with positions where spike waveforms are available (ground truth)
        :param xoffset: Integer for shifting the xpos values
        :return:        Class FrameWaveform with waveforms, labels and position
        """
        return self._frame_generator.frame_generation_with_position(
            xraw=xraw,
            xpos=xpos,
            xoffset=xoffset
        )
