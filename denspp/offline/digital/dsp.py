import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from logging import getLogger
from dataclasses import dataclass
from fxpmath import Fxp
from denspp.offline.analog.common_func import CommonDigitalFunctions
from denspp.offline.plot_helper import save_figure, get_plot_color


@dataclass
class SettingsFilter:
    """Configuration class for defining the filter processor
    Attributes:
        gain:       Integer with applied amplification factor [V/V]
        fs:         Sampling rate [Hz]
        n_order:    Integer with number of filter order
        f_filt:     List with filter frequencies [Hz] (low/high-pass: only one value, band-pass/stop: two values)
        type:       String with selected filter algorithm ['iir', 'fir']
        f_type:     String with selected filter structure ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        b_type:     String with selected filter type ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch' (only IIR)]
        t_dly:      Float with delay time [s] if FIR delay filter is used
        q_fac:      Quality factor (relevant for Notch filter)
    """
    gain: int
    fs: float
    n_order: int
    f_filt: list
    type: str
    f_type: str
    b_type: str
    t_dly: float
    q_fac: float


RecommendedSettingsFilter = SettingsFilter(
    gain=1, fs=0.3e3,
    n_order=2, f_filt=[0.1, 100],
    type='iir', f_type='butter', b_type='bandpass',
    t_dly=100e-6,
    q_fac=10
)


class DSP(CommonDigitalFunctions):
    def __init__(self, setting: SettingsFilter):
        """Class for Emulating Digital Signal Processing on FPGA
        :param setting:     Class for handling the filter stage (using SettingsFilter)
        :return:            None
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.settings = setting

        self.coeff_a = None
        self.coeff_b = None
        self.do_analog = False
        self.use_filtfilt = False
        self.__process_filter()

    @property
    def get_filter_coeffs(self) -> dict:
        """Getting the filter coefficients"""
        return {'b': self.coeff_b.tolist(), 'a': self.coeff_a.tolist()}

    def __process_filter(self) -> None:
        frange = 2 * np.array(self.settings.f_filt) / self.settings.fs
        self._logger.debug(f'Build {self.settings.type.upper()} filter: {self.settings.b_type}, {self.settings.f_type}, {frange}')
        if self.settings.type.lower() == 'iir':
            if not self.settings.b_type.lower() == 'notch':
                filter = scft.iirfilter(
                    N=self.settings.n_order, Wn=frange,
                    ftype=self.settings.f_type.lower(), btype=self.settings.b_type.lower(),
                    analog=self.do_analog, output='ba'
                )
            else:
                filter = scft.iirnotch(
                    w0=float(self.settings.f_filt[0]),
                    Q=self.settings.q_fac,
                    fs=self.settings.fs
                )
            self.coeff_b = filter[0]
            self.coeff_a = filter[1]
        elif self.settings.type.lower() == 'fir':
            if not self.settings.b_type.lower() == 'notch':
                filter = scft.firwin(
                    numtaps=self.settings.n_order, cutoff=frange
                )
                self.coeff_b = filter
                self.coeff_a = np.array(1.0)
            else:
                text = "Change variable - b_type - Notch filter in FIR topology is not available!"
                self._logger.error(text)
                raise ValueError(text)

    def filter(self, xin: np.ndarray) -> np.ndarray:
        """Apply configured filter structure on transient data
        :param xin:     Numpy array with transient data
        :return:        Numpy array with filtered data
        """
        if not self.use_filtfilt:
            xout = self.settings.gain * scft.lfilter(
                b=self.coeff_b,
                a=self.coeff_a,
                x=xin
            ).astype("int")
        else:
            xout = self.settings.gain * scft.filtfilt(
                b=self.coeff_b,
                a=self.coeff_a,
                x=xin
            ).astype("int")
        return xout

    def time_delay_fir(self, xin: np.ndarray) -> np.ndarray:
        """Performing an all-pass filter (FIR) for adding time delay
        :param xin:     Numpy array with transient data
        :return:        Numpy array with delayed data
        """
        set_delay = round(self.settings.t_dly * self.settings.fs)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, xin[0:xin.size - set_delay]), axis=None)
        return uout

    def time_delay_iir_fir_order(self, xin: np.ndarray, f_b: float=1.0, do_plot: bool=False) -> np.ndarray:
        """Performing a 1st order all-pass filter (IIR) for adding time delay
        :param xin:         Numpy array with transient data
        :param f_b:         Filter bandpass filter
        :param do_plot:     Plot figure
        :return:            Numpy array with delayed data
        """
        val = np.tan(np.pi * f_b / self.settings.fs)
        iir_c0 = (val - 1) / (val + 1)

        b = [iir_c0, 1.0]
        a = [1.0, iir_c0]
        if do_plot:
            self.plot_freq_response(b, a)
            self.plot_grp_delay(b, a, show_plot=do_plot)
        return scft.lfilter(b, a, xin)

    def time_delay_iir_sec_order(self, xin: np.ndarray, f_b: float=1.0, bandwidth: float=0.5, do_plot: bool=False) -> np.ndarray:
        """Performing a 2nd order all-pass filter (IIR) for adding time delay
        :param xin:         Numpy array with transient data
        :param f_b:         Filter bandpass filter
        :param bandwidth:   Bandwidth parameter
        :param do_plot:     Plot figure
        :return:            Numpy array with delayed data
        """
        val = np.tan(np.pi * bandwidth / self.settings.fs)
        iir_c0 = (val - 1) / (val + 1)
        iir_c1 = -np.cos(2 * np.pi * f_b / self.settings.fs)

        b = [-iir_c0, iir_c1*(1-iir_c0), 1.0]
        a = [1.0, iir_c1*(1-iir_c0), -iir_c0]
        if do_plot:
            self.plot_freq_response(b, a)
            self.plot_grp_delay(b, a)

        return scft.lfilter(b, a, xin)

    def quantize_coeffs(self, bit_size: int, bit_frac: int, signed: bool=True ) -> dict:
        """Quantize the coefficients with given bit fraction for adding into hardware designs
        :param bit_size:    Integer with total bitwidth
        :param bit_frac:    Integer with fraction width
        :param signed:      Boolean with whether to sign the coefficients
        :return:            Dictionary with quantized coefficients
        """
        self.define_limits(bit_signed=signed, total_bitwidth=bit_size, frac_bitwidth=bit_frac)

        quant_a = Fxp(self.coeff_a, signed=signed, n_word=bit_size, n_frac=bit_frac)
        error_a = self.coeff_a - quant_a.all()
        quant_b = Fxp(self.coeff_b, signed=signed, n_word=bit_size, n_frac=bit_frac)
        error_b = self.coeff_b - quant_b.all()

        return {'b': quant_b, 'a': quant_a, 'error_b': error_b, 'error_a': error_a}

    def coeff_print(self, bit_size: int, bit_frac: int, signed: bool=True) -> None:
        """Printing the coefficients with given bit fraction for adding into hardware designs
        :param bit_size:    Integer with total bitwidth
        :param bit_frac:    Integer with fraction width
        :param signed:      Boolean with whether to sign the coefficients
        """
        print("\nPrinting the filter coeffizients:")
        coeffs_quant = self.quantize_coeffs(bit_size=bit_size, bit_frac=bit_frac, signed=signed)
        if self.coeff_a.size > 1:
            for id, (coeff, error) in enumerate(zip(coeffs_quant['a'], coeffs_quant['error_a'])):
                print(f".. Coeff_A{id}: {float(coeff):.8f} = {coeff.hex()} (Delta = {error:.6f})")

        if self.coeff_b.size > 1:
            for id, (coeff, error) in enumerate(zip(coeffs_quant['b'], coeffs_quant['error_b'])):
                print(f".. Coeff_B{id}: {float(coeff):.8f} = {coeff.hex()} (Delta = {error:.6f})")

    def coeff_verilog(self, bit_size: int, bit_frac: int, signed: bool=True) -> None:
        """Printing the coefficients with given bit fraction for adding into FPGA designs
        :param bit_size:    Integer with total bitwidth
        :param bit_frac:    Integer with fraction width
        :param signed:      Boolean with whether to sign the coefficients
        """
        print(f"\n//--- Used filter coefficients for {self.settings.b_type, self.settings.f_type}")
        coeffs_quant = self.quantize_coeffs(bit_size=bit_size, bit_frac=bit_frac, signed=signed)
        if self.coeff_a.size > 1:
            print(f"wire signed [{bit_size - 1:d}:0] coeff_a [{len(coeffs_quant['a']) - 1:d}:0];")
            for id, coeff in enumerate(coeffs_quant['a']):
                print(f"assign coeff_a[{id}] = {bit_size}'b{coeff.bin(False)}; //coeff_a[{id}] = {float(coeff):.6f} = {coeff.hex()}")

        if self.coeff_b.size > 1:
            print(f"wire signed [{bit_size - 1:d}:0] coeff_b [{len(coeffs_quant['b']) - 1:d}:0];")
            for id, coeff in enumerate(coeffs_quant['b']):
                print(f"assign coeff_b[{id}] = {bit_size}'b{coeff.bin(False)}; //coeff_b[{id}] = {float(coeff):.6f} = {coeff.hex()}")

    def plot_freq_response(self, b: list, a: list, num_points: int=1001,
                           show_plot: bool=True, path2save: str='') -> None:
        """Function for plotting the frequency response of desired filter type
        :param b:           Filter coefficient b
        :param a:           Filter coefficient a
        :param num_points:  Number of points to plot
        :param show_plot:   Boolean for showing plot
        :param path2save:   Path to save figure
        """
        ws = 2 * np.pi * self.settings.fs
        if not len(a) == 0:
            w, h = scft.freqz(b, a, worN=num_points, fs=ws, include_nyquist=True)
        else:
            w, h = scft.freqs(b, 1, worN=num_points, fs=ws)

        f = w / (2 * np.pi)
        # --- Do plotting
        fig1, ax11 = plt.subplots()
        plt.title('Frequency response')
        amplit_log = 20 * np.log10(np.abs(h))
        amplit_ref = np.zeros_like(amplit_log) + amplit_log.max() - 3 * self.settings.n_order
        plt.semilogx(f, amplit_log, color=get_plot_color(0), label='Gain')
        plt.semilogx(f, amplit_ref, linestyle='--', color=get_plot_color(0), label='Ref.')
        plt.ylabel(r'Amplitude |$H(\omega)$| (dB)', color=get_plot_color(0))
        plt.xlabel(r'Frequency $f_\mathrm{sig}$ (Hz)')
        plt.ylim([amplit_log.max()-20, amplit_log.max()+3])
        plt.xlim([f[0], f[-1]])

        ax11.grid(True, which="both", ls="--")
        ax21 = ax11.twinx()

        phase = np.angle(h, deg=True)
        plt.semilogx(f, phase, color=get_plot_color(1), label='Phase')
        plt.ylabel(r'Phase $\alpha$ (°)', color=get_plot_color(1))
        plt.tight_layout()
        if path2save:
            save_figure(plt, path2save, 'freq_response')
        plt.show(block=show_plot)

    def plot_grp_delay(self, b: list, a: list, num_points: int=1001, show_plot: bool=False) -> None:
        """ Plotting the Group Delay of filter """
        ws = 2 * np.pi * self.settings.fs
        if not len(a) == 0:
            w, h = scft.freqz(b, a, worN=num_points, fs=ws, include_nyquist=True)
        else:
            w, h = scft.freqs(b, 1, worN=num_points, fs=ws)

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

    def do_hw_normalization(self, xin: np.ndarray, full_range: bool=False) -> np.ndarray:
        """Normalization of the input to binary shifting, range [+1, -1]
        :param xin:             Numpy array with data
        :param full_range:      If true, normalize to full range
        """
        frame_out = np.zeros_like(xin, dtype='float')
        offset = 0.5 if full_range else 0
        for ite, frame in enumerate(xin):
            max_val = np.max(np.abs(frame))
            scale_val = 1 / np.ceil(np.log2(max_val))
            frame_out[ite, :] = offset + scale_val * frame
        return frame_out
