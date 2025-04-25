import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fxpmath import Fxp
from denspp.offline.analog.common_func import CommonDigitalFunctions


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
        b_type:     String with selected filter type ['lowpass', 'highpass', 'bandpass', 'bandstop']
        t_dly:      Float with delay time [s] if FIR delay filter is used
    """
    gain: int
    fs: float
    n_order: int
    f_filt: list
    type: str
    f_type: str
    b_type: str
    t_dly: float


RecommendedSettingsFilter = SettingsFilter(
    gain=1, fs=0.3e3,
    n_order=2, f_filt=[0.1, 100],
    type='iir', f_type='butter', b_type='bandpass',
    t_dly=100e-6
)


class DSP(CommonDigitalFunctions):
    """Class for Emulating Digital Signal Processing on FPGA"""
    def __init__(self, setting: SettingsFilter):
        super().__init__()
        self.settings = setting

        self.coeff_a = None
        self.coeff_b = None
        self.do_analog = False
        self.use_filtfilt = False
        self.type1 = self.__process_filter()

    def __process_filter(self) -> str:
        type = self.settings.type
        frange = 2 * np.array(self.settings.f_filt) / self.settings.fs
        if type == 'iir':
            # --- Einstellen eines IIR Filters
            filter = scft.iirfilter(
                N=self.settings.n_order, Wn=frange,
                ftype=self.settings.f_type, btype=self.settings.b_type,
                analog=self.do_analog, output='ba'
            )
            self.coeff_b = filter[0]
            self.coeff_a = filter[1]
        elif type == 'fir':
            # --- Einstellen eines FIR Filters
            filter = scft.firwin(
                numtaps=self.settings.n_order, cutoff=frange
            )
            self.coeff_b = filter
            self.coeff_a = np.array(1.0)

        return type

    def filter(self, xin: np.ndarray) -> np.ndarray:
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

    def time_delay_fir(self, uin: np.ndarray) -> np.ndarray:
        """Perfoming an all-pass filter (FIR) for adding time delay"""
        set_delay = round(self.settings.t_dly * self.settings.fs)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
        return uout

    def time_delay_iir_fir_order(self, uin: np.ndarray, f_b: float=1.0, do_plot: bool=False) -> np.ndarray:
        """Performing a 1st order all-pass filter (IIR) for adding time delay"""
        val = np.tan(np.pi * f_b / self.settings.fs)
        iir_c0 = (val - 1) / (val + 1)

        b = [iir_c0, 1.0]
        a = [1.0, iir_c0]
        if do_plot:
            self.plot_freq_response(b, a)
            self.plot_grp_delay(b, a)

        return scft.lfilter(b, a, uin)

    def time_delay_iir_sec_order(self, uin: np.ndarray, f_b: float=1.0, bandwidth: float=0.5,
                                 do_plot: bool=False) -> np.ndarray:
        """Performing a 2nd order all-pass filter (IIR) for adding time delay"""
        val = np.tan(np.pi * bandwidth / self.settings.fs)
        iir_c0 = (val - 1) / (val + 1)
        iir_c1 = -np.cos(2 * np.pi * f_b / self.settings.fs)

        b = [-iir_c0, iir_c1*(1-iir_c0), 1.0]
        a = [1.0, iir_c1*(1-iir_c0), -iir_c0]
        if do_plot:
            self.plot_freq_response(b, a)
            self.plot_grp_delay(b, a)

        return scft.lfilter(b, a, uin)

    def coeff_print(self, bit_size: int, bit_frac: int, signed: bool=True) -> None:
        """Printing the coefficients with given bit fraction for adding into hardware designs"""
        print("\nAusgabe der Filterkoeffizienten:")
        if self.coeff_a:
            for id, coeff in enumerate(self.coeff_a):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                error = coeff - float(quant)
                print(f".. Coeff_A{id}: {float(quant):.8f} = {quant.hex()} (Delta = {error:.6f})")

        if self.coeff_b:
            for id, coeff in enumerate(self.coeff_b):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                error = coeff - float(quant)
                print(f".. Coeff_B{id}: {float(quant):.8f} = {quant.hex()} (Delta = {error:.6f})")

    def coeff_verilog(self, bit_size: int, bit_frac: int, signed: bool=True) -> None:
        """Printing the coefficients with given bit fraction for adding into FPGA designs"""
        self.define_limits(bit_signed=signed, total_bitwidth=bit_size, frac_bitwidth=bit_frac)

        print(f"\n//--- Used filter coefficients for {self.settings.b_type, self.settings.f_type} with {np.array(self.settings.f_filt) / 1000:.3f} kHz @ {self.settings.fs / 1000:.3f} kHz")
        if not self.type1 == 'fir':
            coeffa_size = len(self.coeff_a)
            print(f"wire signed [{bit_size - 1:d}:0] coeff_a [{coeffa_size - 1:d}:0];")
            for id, coeff in enumerate(self.coeff_a):
                quant = Fxp(-coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                print(f"assign coeff_a[{id}] = {bit_size}'b{quant.bin(False)}; //coeff_a[{id}] = {float(quant):.6f} = {quant.hex()}")

        coeffb_size = len(self.coeff_b)
        print(f"wire signed [{bit_size - 1:d}:0] coeff_b [{coeffb_size - 1:d}:0];")
        for id, coeff in enumerate(self.coeff_b):
            quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
            print(f"assign coeff_b[{id}] = {bit_size}'b{quant.bin(False)}; //coeff_b[{id}] = {float(quant):.6f} = {quant.hex()}")

    def plot_freq_response(self, b: list, a: list, num_points: int=1001) -> None:
        ws = 2 * np.pi * self.settings.fs
        if not len(a) == 0:
            w, h = scft.freqz(b, a, worN=num_points, fs=ws, include_nyquist=True)
        else:
            w, h = scft.freqs(b, 1, worN=num_points, fs=ws)

        f = w / (2 * np.pi)
        # --- Do plotting
        fig1, ax11 = plt.subplots()
        plt.title('Frequency response')
        plt.semilogx(f, 20 * np.log10(abs(h)), 'b')
        plt.ylabel(r'Amplitude |$H(\omega)$| (dB)', color='b')
        plt.xlabel(r'Frequency $f_\mathrm{sig}$ (Hz)')
        plt.ylim([-0.5, 0.5])

        ax11.grid()
        ax21 = ax11.twinx()

        phase = np.unwrap(np.angle(h)) / np.pi * 180
        plt.semilogx(f, phase, 'g')
        plt.ylabel(r'Phase $\alpha$ (°)', color='g')
        plt.tight_layout()

    def plot_grp_delay(self, b: list, a: list, num_points: int=1001) -> None:
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

    def do_hw_normalization(self, input: np.ndarray, full_range: bool=False) -> np.ndarray:
        """Normalization of the input to binary shifting, range [+1, -1]"""
        frame_out = np.zeros(shape=input.shape, dtype='float')
        offset = 0.5 if full_range else 0
        for ite, frame in enumerate(input):
            max_val = np.max(np.abs(frame))
            scale_val = 1 / np.ceil(np.log2(max_val))
            frame_out[ite, :] = offset + scale_val * frame

        return frame_out
