import numpy as np
import scipy.signal as scft
import matplotlib.pyplot as plt
from fxpmath import Fxp

# TODO: Filtertyp außer 'butter' noch einfügen
class filter_stage():
    # btype = ['how', 'high', 'bandpass', 'bandstop', 'notch', 'peak']
    # ftype = ['iir', 'fir']
    def __init__(self, N: int, fs: int, f_filter: int, btype: str, ftype: str):
        self.n_order = N
        self.fs = fs
        self.f_filter = np.array(f_filter, dtype='float')
        self.f_coeff = 2 * self.f_filter / self.fs

        self.coeff_a = None
        self.coeff_b = None

        self.do_analog = False
        self.use_filtfilt = False
        self.type_iir = 'butter'
        self.type0 = self.__process_type(btype)
        self.type1 = self.__process_filter(ftype)

        self.value = 0.4995 #2 * math.pi
        self.value_quant = Fxp(self.value, signed=True, n_word=8, n_frac=8)

    def filter(self, xin: np.ndarray) -> np.ndarray:
        if not self.use_filtfilt:
            xout = scft.lfilter(
                b=self.coeff_b,
                a=self.coeff_a,
                x=xin
            )
        else:
            xout = scft.filtfilt(
                b=self.coeff_b,
                a=self.coeff_a,
                x=xin
            )

        return xout

    def coeff_print(self, bit_size: int, bit_frac: int):
        signed = True
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


    def coeff_verilog(self, bit_size: int, bit_frac: int):
        signed = True
        print(f"\n//--- Used filter coefficients for {self.type_iir, self.type0} with {self.f_filter / 1000:.3f} kHz @ {self.fs / 1000:.3f} kHz")
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

    def freq_response(self, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.type1 == 'iir':
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
        elif self.type1 == 'fir':
            w, h = scft.freqz(
                b=self.coeff_b, a=1,
                fs=20000, worN=f0
            )

        h0 = np.array(h)
        gain = np.abs(h0)
        phase = np.angle(h, deg=True)

        return w, gain, phase

    def __process_filter(self, type: str) -> str:
        type0 = str()
        if(type == 'iir'):
            type0 = 'iir'
            # --- Einstellen eines IIR Filters
            filter = scft.iirfilter(
                N=self.n_order, Wn=self.f_coeff,
                btype=self.type0, ftype='butter',
                analog=self.do_analog, output='ba'
            )
            self.coeff_b = filter[0]
            self.coeff_a = filter[1]
        elif(type == 'fir'):
            type0 = 'fir'
            # --- Einstellen eines FIR Filters
            filter = scft.firwin(
                numtaps=self.n_order, cutoff=self.f_coeff
            )
            self.coeff_b = filter
            self.coeff_a = np.array(1.0)

        return type0
    def __process_type(self, type: str) -> str:
        type0 = str()
        if(type == 'low'):
            type0 = 'lowpass'
        elif(type == 'high'):
            type0 = 'highpass'
        elif(type == 'bandpass'):
            type0 = 'bandpass'
        elif(type == 'bandstop'):
            type0 = 'bandstop'

        return type0

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
        filter = filter_stage(Nfilt, fs, ftp, 'high', 'iir')
        coeff_a.append(filter.coeff_a)
        coeff_asum.append(np.sum(filter.coeff_a))
        coeff_b.append(filter.coeff_b)
        coeff_bsum.append(np.sum(filter.coeff_a))

    coeff_a = np.array(coeff_a)
    coeff_asum = np.array(coeff_asum)
    coeff_b = np.array(coeff_b)
    coeff_bsum = np.array(coeff_bsum)

    # --- Plotten
    plt.figure()
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212)

    ax0.plot(f_low/fs, coeff_a)
    ax0.plot(f_low / fs, coeff_asum, color='k')
    ax0.set_ylabel('coeff_a')

    ax1.plot(f_low/fs, coeff_b)
    ax1.plot(f_low / fs, coeff_bsum, color='k')
    ax1.set_ylabel('coeff_b')
    ax1.set_xlabel('f_TP/fs')

    plt.tight_layout(pad=0.5)
    plt.show(block = True)
