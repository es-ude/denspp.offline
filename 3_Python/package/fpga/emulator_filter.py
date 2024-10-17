import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from fxpmath import Fxp


# TODO: Filtertyp außer 'butter' noch einfügen
class filter_stage:
    __filter_type = {'low': 'lowpass', 'high': 'highpass', 'all': 'allpass', 'bandpass': 'bandpass',
                   'bandstop': 'bandstop', 'notch': 'notch'}
    __coeff_a: np.ndarray
    __coeff_b: np.ndarray

    def __init__(self, N: int, fs: float, f_filter: list, btype: str, use_iir_filter: bool):
        """Class for filtering and getting the filter coefficient
        Args:
            N:                  Order number of used filter
            fs:                 Sampling rate of data stream
            f_filter:           Filter coefficient
            btype:              Used filter type ['low', 'high', 'bandpass', 'bandstop', 'notch', 'all']
            use_iir_filter:     Used filter topology [True: IIR, False: FIR]
        """
        self.n_order = N
        self.fs = fs
        self.f_filter = np.array(f_filter, dtype='float')
        self.f_coeff = 2 * self.f_filter / self.fs
        self.__coeffb_defined = False
        self.__coeffa_defined = False
        self.do_analog = False
        self.use_filtfilt = False
        self.type_iir_used = use_iir_filter
        self.type_iir = 'butter'
        self.type0 = self.__process_type(btype)
        self.__process_filter()

    def filter(self, xin: np.ndarray) -> np.ndarray:
        """Performing the filtering of input data
        Args:
            xin:    Input numpy array
        Returns:
            Numpy array with filtered output
        """
        if not self.use_filtfilt:
            xout = scft.lfilter(
                b=self.__coeff_b,
                a=self.__coeff_a,
                x=xin
            )
        else:
            xout = scft.filtfilt(
                b=self.__coeff_b,
                a=self.__coeff_a,
                x=xin
            )

        return xout

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

        return {'coeffa': coeffa, 'coeffa_error': coeffa_error, 'coeffb': coeffb, 'coeffb_error': coeffb_error}

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
        print(f"\n//--- Used filter coefficients for {self.type_iir, self.type0} with {self.f_filter / 1000:.3f} kHz @ {self.fs / 1000:.3f} kHz")
        if self.type_iir_used:
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

    def freq_response(self, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Getting the frequency response of the filter for specific frequency values
        Args:
            f0:     Numpy array with frequency values
        Return:
            Tuple with [frequency, gain, phase]
        """
        if self.type_iir_used:
            fout = scft.iirfilter(
                N=self.n_order, Wn=self.f_filter, #rs=60, rp=1,
                btype=self.type0, ftype=self.type_iir, analog=True,
                output='ba'
            )
            b = fout[0]
            a = fout[1]

            w, h = scft.freqs(
                b=b, a=a,
                worN=f0
            )
        else:
            w, h = scft.freqz(
                b=self.__coeff_b, a=1,
                fs=20000, worN=f0
            )

        # --- Output
        h0 = np.array(h)
        gain = np.abs(h0)
        phase = np.angle(h, deg=True)
        return w, gain, phase

    def __process_filter(self) -> None:
        """Extracting the filter coefficient with used settings"""
        if self.type_iir_used:
            # --- Einstellen eines IIR Filters
            filter_params = scft.iirfilter(
                N=self.n_order, Wn=self.f_coeff,
                btype=self.type0, ftype='butter',
                analog=self.do_analog, output='ba'
            )
            self.__coeff_b = filter_params[0]
            self.__coeffb_defined = True
            self.__coeff_a = filter_params[1]
            self.__coeffa_defined = True
        else:
            # --- Einstellen eines FIR Filters
            filter_params = scft.firwin(
                numtaps=self.n_order, cutoff=self.f_coeff
            )
            self.__coeff_b = filter_params
            self.__coeffb_defined = True
            self.__coeff_a = np.array(1.0)
            self.__coeffa_defined = False

    def __process_type(self, type: str) -> str:
        """Definition of the filter type used in scipy function"""
        type_out = ''
        for key, type0 in self.__filter_type.items():
            if type is key:
                type_out = type0
                break
        if type_out == '':
            raise NotImplementedError("Type of used filter type is not available")
        return type_out


if __name__ == "__main__":
    fs = 100
    f_low = np.arange(0.1, 50, 0.1)
    Nfilt = 2

    N = 12
    Ndec = 2
    dec = 2**(Ndec-1)
    lsb = 1/(2**(N-Ndec)-1)
    print(f"Kleinste Dezimalstellung: +/-{dec} und Kommastelle: {lsb}")

    # --- Koeffizienten herleiten
    coeff_a = []
    coeff_asum = []
    coeff_b = []
    coeff_bsum = []
    for i, ftp in enumerate(f_low):
        filter = filter_stage(Nfilt, fs, ftp, 'high', True)

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
