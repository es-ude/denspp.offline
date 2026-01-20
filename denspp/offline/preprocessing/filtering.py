import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from logging import getLogger, Logger
from dataclasses import dataclass
from fxpmath import Fxp
from denspp.offline.analog.common_func import CommonDigitalFunctions
from denspp.offline.plot_helper import save_figure, get_plot_color, get_textsize_paper


@dataclass
class FilterCoeffs:
    """Dataclass with filter coefficients"""
    a: list
    b: list


@dataclass
class SettingsFilter:
    """Configuration class for defining the filter processor
    Attributes:
        gain:       Integer with applied amplification factor [V/V]
        fs:         Sampling rate [Hz]
        n_order:    Integer with number of filter order or delay values in FIR-allpass
        f_filt:     List with filter frequencies [Hz] (low/high-pass: only one value - rest: two values)
        type:       String with selected filter algorithm ['iir', 'fir']
        f_type:     String with selected filter structure ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        b_type:     String with selected filter type ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch', 'allpass']
    """
    gain: int
    fs: float
    n_order: int | float
    f_filt: list
    type: str
    f_type: str
    b_type: str


DefaultSettingsFilter = SettingsFilter(
    gain=1, fs=0.3e3,
    n_order=2, f_filt=[0.1, 100],
    type='iir', f_type='butter', b_type='bandpass'
)


class Filtering(CommonDigitalFunctions):
    __logger: Logger
    _type_supported: list = ['fir', 'iir']
    _btype_supported: list = ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch', 'allpass']
    _ftype_supported: list = ['butter', 'bessel', 'cheby1', 'cheby2', 'ellip']
    _coeff_a: np.ndarray
    _coeff_b: np.ndarray
    _settings: SettingsFilter

    def __init__(self, setting: SettingsFilter, use_filtfilt: bool=False):
        """Class for Emulating Digital Signal Processing on FPGA
        :param setting:         Class for handling the filter stage (using SettingsFilter)
        :param use_filtfilt:    Boolean for applying zero-phase filtering
        :return:                None
        """
        super().__init__()
        self.__logger = getLogger(__name__)
        self._settings = setting
        self.__use_filtfilt = use_filtfilt
        self.__process_filter()

    def get_coeffs(self) -> FilterCoeffs:
        """Getting the filter coefficients"""
        return FilterCoeffs(
            b=self._coeff_b.tolist(),
            a=self._coeff_a.tolist(),
        )

    def get_coeffs_quantized(self, bit_size: int, bit_frac: int, signed: bool=True) -> tuple[FilterCoeffs, dict]:
        """Quantize the coefficients with given bit fraction for adding into hardware designs
        :param bit_size:    Integer with total bitwidth
        :param bit_frac:    Integer with fraction width
        :param signed:      Boolean with whether to sign the coefficients
        :return:            Dictionary with quantized coefficients
        """
        self.define_limits(bit_signed=signed, total_bitwidth=bit_size, frac_bitwidth=bit_frac)
        quant_a = Fxp(self._coeff_a, signed=signed, n_word=bit_size, n_frac=bit_frac)
        error_a = self._coeff_a - quant_a.all()
        quant_b = Fxp(self._coeff_b, signed=signed, n_word=bit_size, n_frac=bit_frac)
        error_b = self._coeff_b - quant_b.all()
        return FilterCoeffs(
            b=quant_b.tolist(),
            a=quant_a.tolist(),
        ), {'b': error_b, 'a': error_a}

    def __extract_filter_coeffs_iir(self) -> None:
        frange = np.array(self._settings.f_filt)
        match self._settings.b_type.lower():
            case 'notch':
                filter = scft.iirnotch(
                    w0=float(self._settings.f_filt[0]),
                    Q=self._settings.n_order,
                    fs=self._settings.fs
                )
                self._coeff_b = filter[0]
                self._coeff_a = filter[1]
            case 'allpass':
                if self._settings.n_order == 1:
                    assert len(self._settings.f_filt) == 1, "f_filt should have length of 1 with [f_b] value"
                    val = np.tan(np.pi * frange[0] / self._settings.fs)
                    iir_c0 = (val - 1) / (val + 1)
                    self._coeff_b = np.array([iir_c0, 1.0])
                    self._coeff_a = np.array([1.0, iir_c0])
                elif self._settings.n_order == 2:
                    assert len(self._settings.f_filt) == 2, "f_filt should have length of 2 with [f_b, bandwidth] value"
                    val = np.tan(np.pi * frange[1] / self._settings.fs)
                    iir_c0 = (val - 1) / (val + 1)
                    iir_c1 = -np.cos(2 * np.pi * frange[0] / self._settings.fs)
                    self._coeff_b = np.array([-iir_c0, iir_c1 * (1 - iir_c0), 1.0])
                    self._coeff_a = np.array([1.0, iir_c1 * (1 - iir_c0), -iir_c0])
                else:
                    raise NotImplementedError
            case _:
                filter = scft.iirfilter(
                    N=self._settings.n_order,
                    Wn=frange[0] if len(frange) == 1 else frange,
                    fs=self._settings.fs,
                    ftype=self._settings.f_type.lower(),
                    btype=self._settings.b_type.lower(),
                    analog=False,
                    output='ba'
                )
                self._coeff_b = filter[0]
                self._coeff_a = filter[1]

    def __extract_filter_coeffs_fir(self) -> None:
        frange = np.array(self._settings.f_filt)
        self._coeff_a = np.array(1.0)
        match self._settings.b_type.lower():
            case 'notch':
                assert len(self._settings.f_filt) == 2, "Size of f_filt should be 2 with [f_notch, bandwidth]"
                freq = [0, frange[0] - frange[1], frange[0], frange[0] + frange[1], self._settings.fs / 2]
                gain = [1, 1, 0, 1, 1]

                self._coeff_b = scft.firwin2(
                    numtaps=self._settings.n_order,
                    freq=freq,
                    gain=gain,
                    fs=self._settings.fs
                )
            case 'allpass':
                self._coeff_b = self._coeff_a
            case _:
                self._coeff_a = np.array(1.0)
                self._coeff_b = scft.firwin(
                    numtaps=self._settings.n_order,
                    cutoff=frange,
                    fs=self._settings.fs,
                    pass_zero=self._settings.b_type.lower()
                )

    def __process_filter(self) -> None:
        assert self._settings.type.lower() in self._type_supported, f"Type {self._settings.type} is not supported from {self._type_supported}"
        assert self._settings.f_type.lower() in self._ftype_supported, f"Filter type {self._settings.f_type} is not supported from {self._ftype_supported}"
        assert self._settings.b_type.lower() in self._btype_supported, f"Structure type {self._settings.b_type} is not supported from {self._btype_supported}"
        self.__logger.debug(f'Build {self._settings.type.upper()} filter: {self._settings.b_type}, {self._settings.f_type}')

        if self._settings.type.lower() == 'iir':
            self.__extract_filter_coeffs_iir()
        elif self._settings.type.lower() == 'fir':
            self.__extract_filter_coeffs_fir()

    def filter(self, xin: np.ndarray) -> np.ndarray:
        """Apply configured filter structure on transient data
        :param xin:     Numpy array with transient data
        :return:        Numpy array with filtered data
        """
        if self._settings.type.lower() == 'fir' and self._settings.b_type.lower() == 'allpass':
            mat = np.zeros(shape=(self._settings.n_order,), dtype=float)
            xout = np.concatenate((mat, xin[0:xin.size - self._settings.n_order]), axis=None)
        elif not self.__use_filtfilt:
            xout = self._settings.gain * scft.lfilter(
                b=self._coeff_b,
                a=self._coeff_a,
                x=xin
            )
        else:
            xout = self._settings.gain * scft.filtfilt(
                b=self._coeff_b,
                a=self._coeff_a,
                x=xin
            )
        return xout

    def __get_frequency_behaviour(self, num_points: int=1001) -> tuple[np.ndarray, np.ndarray]:
        if self._settings.type == 'iir':
            frange = np.array(self._settings.f_filt)
            filter = scft.iirfilter(
                N=self._settings.n_order,
                Wn=frange[0] if len(frange) == 1 else frange,
                ftype=self._settings.f_type.lower(),
                btype=self._settings.b_type.lower(),
                analog=True,
                output='ba'
            )
            return scft.freqs(
                b=filter[0],
                a=filter[1],
                worN=num_points
            )
        else:
            return scft.freqs(
                b=self._coeff_b,
                a=1,
                worN=num_points
            )

    def plot_freq_response(self, num_points: int=1001,
                           show_plot: bool=True, path2save: str='') -> None:
        """Function for plotting the frequency response of desired filter type
        :param num_points:  Number of points to plot
        :param show_plot:   Boolean for showing plot
        :param path2save:   Path to save figure
        """
        w, h = self.__get_frequency_behaviour(num_points=num_points)
        f = w / (2 * np.pi) if self._settings.f_filt == 'fir' else w

        fig1, ax11 = plt.subplots()
        plt.title('Frequency response')
        amplit_log = 20 * np.log10(np.abs(h))
        plt.semilogx(f, amplit_log, color=get_plot_color(0), label='Gain')
        plt.ylabel(r'Amplitude |$H(\omega)$| (dB)', size=get_textsize_paper(), color=get_plot_color(0))
        plt.xlabel(r'Frequency $f_\mathrm{sig}$ (Hz)', size=get_textsize_paper())
        plt.xlim([f[0], f[-1]])

        ax11.grid(True, which="both", ls="--")
        ax21 = ax11.twinx()

        phase = np.angle(h, deg=True)
        plt.semilogx(f, phase, color=get_plot_color(1), label='Phase', alpha=.6)
        plt.ylabel(r'Phase $\alpha$ (°)', size=get_textsize_paper(), color=get_plot_color(1))
        plt.tight_layout()
        if path2save:
            save_figure(plt, path2save, 'freq_response')
        plt.show(block=show_plot)

    def plot_grp_delay(self, num_points: int=1001, show_plot: bool=False) -> None:
        """ Plotting the Group Delay of filter
        :param num_points:  Number of points to plot
        :param show_plot:   Boolean for showing plot
        :return:            None
        """
        w, h = self.__get_frequency_behaviour(num_points=num_points)
        f = w / (2 * np.pi)
        phase = np.unwrap(np.angle(h)) / np.pi * 180
        grp_dly = - np.diff(phase) / np.diff(w)

        plt.figure()
        plt.semilogx(f[2:], grp_dly[1:], 'k', linewidth=1)
        plt.ylabel(r'Group Delay $\tau_\mathrm{grp}$ (s)')
        plt.xlabel(r'Frequency $f_\mathrm{sig}$ (Hz)')
        plt.grid()
        plt.tight_layout()
        plt.show(block=show_plot)
