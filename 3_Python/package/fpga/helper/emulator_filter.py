import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from fxpmath import Fxp
from package.plot.plot_common import save_figure


class filter_stage:
    __filt_btype = {'low': 'lowpass', 'high': 'highpass', 'bandpass': 'bandpass',
                    'bandstop': 'bandstop', 'all': 'allpass'}
    __filt_ftype = {'butter': 'butter', 'cheby1': 'cheby1', 'cheby2': 'cheby2',
                    'ellip': 'ellip', 'bessel': 'bessel'}
    __coeff_a: np.ndarray
    __coeff_b: np.ndarray

    def __init__(self, N: int, fs: float, f_filter: list, use_iir_filter: bool,
                 btype='low', ftype='butter', use_filtfilt=False):
        """Class for filtering and getting the filter coefficient
        (If using Allpass
        Args:
            N:                  Order number of used filter
            fs:                 Sampling rate of data stream
            f_filter:           Filter corner frequency as list ('low', 'high', 'all' (1st order) = [f_brk]
                                and 'bandpass', 'bandstop' = [f_c0, f_c1] and 'all' (2nd order)' = [f_brk and f_bw])
            btype:              Used filter type ['low', 'high', 'bandpass', 'bandstop', 'all' (IIR, 1st/2nd Order)]
            ftype:              Used filter design ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
            use_iir_filter:     Used filter topology [True: IIR, False: FIR]
            use_filtfilt:       Using filtfilt functionality from scipy.signal (Zero-phase filtering)
        """
        self.__sampling_rate = fs
        self.__coeffb_defined = False
        self.__coeffa_defined = False
        self.__use_filtfilt = use_filtfilt

        self.__filter_type_iir_used = use_iir_filter
        self.__filter_order = N
        self.__filter_corner = np.array(f_filter, dtype='float')
        self.__filt_ftype = self.__get_filter_ftype(ftype)
        self.__filt_btype = self.__get_filter_btype(btype)
        self.__extract_filter_params()

    def __extract_filter_params(self) -> None:
        """Extracting the filter coefficient with used settings"""
        if self.__filter_type_iir_used and not self.__filt_btype == 'allpass':
            # --- Defining an IIR filter (excluding allpass)
            filter_params = scft.iirfilter(
                N=self.__filter_order, Wn=2 * self.__filter_corner / self.__sampling_rate,
                btype=self.__filt_btype, ftype=self.__filt_ftype,
                analog=False, output='ba'
            )
            self.__coeff_b = filter_params[0]
            self.__coeffb_defined = True
            self.__coeff_a = np.array(filter_params[1])
            self.__coeffa_defined = True
        elif self.__filter_type_iir_used and self.__filt_btype == 'allpass':
            match self.__filter_order:
                case 1:
                    # --- Getting the coefficient (First Order)
                    val = np.tan(np.pi * self.__filter_corner[0] / self.__sampling_rate)
                    iir_c0 = (val - 1) / (val + 1)
                    self.__coeff_b = np.array([iir_c0, 1.0])
                    self.__coeffb_defined = True
                    self.__coeff_a = np.array([1.0, iir_c0])
                    self.__coeffa_defined = True
                case 2:
                    # --- Getting the coefficient (Second Order)
                    val = np.tan(np.pi * self.__filter_corner[1] / self.__sampling_rate)
                    iir_c0 = (val - 1) / (val + 1)
                    iir_c1 = -np.cos(2 * np.pi * self.__filter_corner[0] / self.__sampling_rate)
                    self.__coeff_b = np.array([-iir_c0, iir_c1 * (1 - iir_c0), 1.0])
                    self.__coeffb_defined = True
                    self.__coeff_a = np.array([1.0, iir_c1 * (1 - iir_c0), -iir_c0])
                    self.__coeffa_defined = True
                case _:
                    raise NotImplementedError("Allpass IIR-filters are only implemented for 1st and 2nd order! "
                                              "- Please change!")
        elif self.__filter_type_iir_used and self.__filt_btype == 'allpass':
            raise NotImplementedError("Allpass Filter is only implemented for IIR filter types!")
        else:
            # --- Defining a FIR filter
            filter_params = scft.firwin(
                numtaps=self.__filter_order,
                cutoff=self.__filter_corner, fs=self.__sampling_rate,
                pass_zero=self.__filt_btype
            )
            self.__coeff_b = filter_params
            self.__coeffb_defined = True
            self.__coeff_a = np.array(1.0)
            self.__coeffa_defined = False

    def __get_filter_btype(self, type_used: str) -> str:
        """Definition of the filter type used in scipy function"""
        type_out = ''
        for key, type0 in self.__filt_btype.items():
            if type_used is key:
                type_out = type0
                break
        if type_out == '':
            raise NotImplementedError("Type of used filter type is not available")
        return type_out

    def __get_filter_ftype(self, type_used: str) -> str:
        """Definition of the filter type used in scipy function"""
        type_out = ''
        for key, type0 in self.__filt_ftype.items():
            if type_used is key:
                type_out = type0
                break
        if type_out == '':
            raise NotImplementedError("Type of used filter type is not available")
        return type_out

    def filter(self, xin: np.ndarray) -> np.ndarray:
        """Performing the filtering of input data
        Args:
            xin:    Input numpy array
        Returns:
            Numpy array with filtered output
        """
        return scft.lfilter(b=self.__coeff_b, a=self.__coeff_a, x=xin) if not self.__use_filtfilt \
            else scft.filtfilt(b=self.__coeff_b, a=self.__coeff_a, x=xin)

    def coeff_print(self, bit_size: int, bit_frac: int, signed=True) -> None:
        """Printing the filter coefficient in quantized matter
        Args:
            bit_size:   Bitwidth of the data in total
            bit_frac:   Bitwidth of fraction
            signed:     Option if data type is signed (True) or unsigned (False)
        Return:
            None
        """
        print("\nAusgabe der Filterkoeffizienten:")
        if self.__coeffa_defined:
            for id, coeff in enumerate(self.__coeff_a):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                error = coeff - float(quant)
                print(f".. Coeff_A{id}: {float(quant):.8f} = {quant.hex()} (Delta = {error:.6f})")

        if self.__coeffb_defined:
            for id, coeff in enumerate(self.__coeff_b):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                error = coeff - float(quant)
                print(f".. Coeff_B{id}: {float(quant):.8f} = {quant.hex()} (Delta = {error:.6f})")

    def get_coeff_quant(self, bit_size: int, bit_frac: int, signed=True) -> dict:
        """Getting the filter coefficient in quantized matter
        Args:
            bit_size:   Bitwidth of the data in total
            bit_frac:   Bitwidth of fraction
            signed:     Option if data type is signed (True) or unsigned (False)
        Return:
            Dict with filter coefficients
        """
        coeffa = list()
        coeffa_error = list()
        coeffb = list()
        coeffb_error = list()

        if self.__coeffa_defined:
            for id, coeff in enumerate(self.__coeff_a):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                coeffa.append(quant)
                coeffa_error.append(coeff - float(quant))

        if self.__coeffb_defined:
            for id, coeff in enumerate(self.__coeff_b):
                quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                coeffb.append(quant)
                coeffb_error.append(coeff - float(quant))

        return {'coeffa': coeffa, 'coeffa_error': coeffa_error, 'coeffb': coeffb, 'coeffb_error': coeffb_error}

    def get_coeff_full(self) -> dict:
        """Getting the filter coefficient in quantized matter
        Args:
            None
        Return:
            Dict with filter coefficients
        """
        coeffa = list()
        coeffa_error = list()
        coeffb = list()
        coeffb_error = list()

        if self.__coeffa_defined:
            coeffa = self.__coeff_a.tolist()
            coeffa_error.append([0.0 for idx in self.__coeff_a])

        if self.__coeffb_defined:
            coeffb = self.__coeff_b.tolist()
            coeffb_error.append([0.0 for idx in self.__coeff_b])

        return {'coeffa': np.array(coeffa), 'coeffa_error': np.array(coeffa_error),
                'coeffb': np.array(coeffb), 'coeffb_error': np.array(coeffb_error)}

    def get_coeff_verilog(self, bit_size: int, bit_frac: int, signed=True, do_print=False) -> list:
        """Printing the filter coefficient for Verilog in quantized matter
        Args:
            bit_size:   Bitwidth of the data in total
            bit_frac:   Bitwidth of fraction
            signed:     Option if data type is signed (True) or unsigned (False)
            do_print:   Print the code
        Return:
            List with string output
        """
        print_out = list()
        print(f"\n//--- Used filter coefficients for {self.__filt_ftype, self.__filt_btype} with "
              f"{self.__filter_corner / 1000:.3f} kHz @ {self.__sampling_rate / 1000:.3f} kHz")
        if self.__filter_type_iir_used:
            coeffa_size = len(self.__coeff_a)
            print_out.append(f"wire signed [{bit_size - 1:d}:0] coeff_a [{coeffa_size - 1:d}:0];")
            for id, coeff in enumerate(self.__coeff_a):
                quant = Fxp(-coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
                print_out.append(f"assign coeff_a[{id}] = {bit_size}'b{quant.bin(False)}; "
                                 f"//coeff_a[{id}] = {float(quant):.6f} = {quant.hex()}")
            print_out.append('\fn')

        coeffb_size = len(self.__coeff_b)
        print_out.append(f"wire signed [{bit_size - 1:d}:0] coeff_b [{coeffb_size - 1:d}:0];")
        for id, coeff in enumerate(self.__coeff_b):
            quant = Fxp(coeff, signed=signed, n_word=bit_size, n_frac=bit_frac)
            print_out.append(f"assign coeff_b[{id}] = {bit_size}'b{quant.bin(False)}; "
                             f"//coeff_b[{id}] = {float(quant):.6f} = {quant.hex()}")

        # --- Generate output
        if do_print:
            for line in print_out:
                print(line)
        return print_out

    def freq_response(self, freq: np.ndarray, do_plot=False) -> dict:
        """Getting the frequency response of the filter for specific frequency values
        Args:
            freq:       Numpy array with frequency values
            do_plot:    Plotting the results
        Return:
            Dict with ['freq': frequency, 'gain': gain, 'phase': phase]
        """
        if self.__filter_type_iir_used:
            fout = scft.iirfilter(
                N=self.__filter_order, Wn=self.__filter_corner,
                btype=self.__filt_btype, ftype=self.__filt_ftype, analog=True,
                output='ba'
            )
            w, h = scft.freqs(b=fout[0], a=fout[1], worN=freq)
        else:
            w, h = scft.freqz(b=self.__coeff_b, a=1, fs=self.__sampling_rate, worN=freq)

        h0 = np.array(h)
        gain = np.abs(h0)
        phase = np.angle(h0, deg=True)

        # --- Figure generation
        if do_plot:
            plot_bode_diagramm(w, gain, phase)
        return {'freq': w, 'gain': gain, 'phase': phase}


def plot_bode_diagramm(freq: np.ndarray, gain: np.ndarray, phase: np.ndarray, path2save='') -> None:
    """Plotting the Bode diagramm of defined filter characteristics
    Args:
        freq:       Numpy array of used frequency
        gain:       Numpy array of available gain from filter
        phase:      Numpy array of available phase from filter
        path2save:  Optional string for plotting [Default: '' for non-plotting]
    Return:
        None
    """
    axs = plt.subplots(2, 1, sharex="all")[1]
    axs[0].semilogx(freq, 20 * np.log10(np.array(gain)), 'k', linewidth=1, marker='.', label="Gain")
    axs[0].set_ylabel(r"Gain $v_U$ / dB")
    axs[0].set_xlim([freq[0], freq[-1]])

    axs[1].semilogx(freq, np.array(phase), 'r', linewidth=1, marker='.', label="Phase")
    axs[1].set_ylabel(r"Phase $\alpha$ / °")
    axs[1].set_xlabel(r'Frequency $f$ / Hz')
    axs[1].set_xlim([freq[0], freq[-1]])

    for ax in axs:
        ax.set_xlim([freq[0], freq[-1]])
        ax.grid(which='both', linestyle='--')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    if path2save:
        save_figure(plt, path2save, 'filter_charac_bode')
    plt.show(block=True)


if __name__ == "__main__":
    fs = 100
    freq = np.logspace(-2, 2, 101)
    f_low = np.arange(1, 50, 0.1)
    Nfilt = 2001

    # --- Koeffizienten herleiten
    coeff_a = []
    coeff_asum = []
    coeff_b = []
    coeff_bsum = []
    for i, ftp in enumerate(f_low):
        filter = filter_stage(Nfilt, fs, [ftp], False, 'high')
        filter.freq_response(freq, True)
        coeff = filter.get_coeff_full()

        coeff_a.append(coeff['coeffa'])
        coeff_asum.append(np.sum(coeff['coeffa']))
        coeff_b.append(coeff['coeffb'])
        coeff_bsum.append(np.sum(coeff['coeffb']))

    coeff_a = np.array(coeff_a)
    coeff_asum = np.array(coeff_asum)
    coeff_b = np.array(coeff_b)
    coeff_bsum = np.array(coeff_bsum)

    # --- Plotten
    axs = plt.subplots(2, 1, sharex=True)[1]

    axs[0].plot(f_low/fs, coeff_a)
    axs[0].plot(f_low / fs, coeff_asum, color='k')
    axs[0].set_ylabel('coeff_a')
    axs[0].grid()

    axs[1].plot(f_low/fs, coeff_b)
    axs[1].plot(f_low / fs, coeff_bsum, color='k')
    axs[1].set_ylabel('coeff_b')
    axs[1].set_xlabel('f_TP/fs')
    axs[1].grid()
    axs[1].set_xlim([0.0, 0.5])

    plt.tight_layout()
    plt.show(block=True)
