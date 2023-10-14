import dataclasses
import numpy as np
import scipy.signal as scft
from fxpmath import Fxp


@dataclasses.dataclass
class SettingsDSP:
    gain: int
    fs: float
    n_order: int
    f_filt: list
    type: str           # type = ['iir', 'fir']
    f_type: str         # f_type = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    b_type: str         # btype = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    t_dly: float


@dataclasses.dataclass
class RecommendedSettingsDSP(SettingsDSP):
    def __init__(self):
        super().__init__(
            gain=1, fs=20e3,
            n_order=2, f_filt=[0.1, 100],
            type='iir', f_type='butter', b_type='bandpass',
            t_dly=100e-6
        )


class DSP:
    """Class for Emulating Digital Signal Processing on FPGA"""
    def __init__(self, setting: SettingsDSP):
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

    def time_delay(self, uin: np.ndarray) -> np.ndarray:
        set_delay = round(self.settings.t_dly * self.settings.fs)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
        return uout

    def coeff_print(self, bit_size: int, bit_frac: int, signed=True) -> None:
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

    def coeff_verilog(self, bit_size: int, bit_frac: int, signed=True) -> None:
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

    def freq_response(self, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frange = 2 * np.array(self.settings.f_filt) / self.settings.fs
        if self.type1 == 'iir':
            fout = scft.iirfilter(
                N=self.settings.n_order, Wn=frange, #rs=60, rp=1,
                btype=self.settings.f_type, ftype=self.settings.b_type, analog=True,
                output='ba'
            )
            b = fout[0]
            a = fout[1]

            w, h = scft.freqs(
                b=b, a=a,
                worN=f0
            )
        else:  #self.type1 == 'fir':
            w, h = scft.freqz(
                b=self.coeff_b, a=1,
                fs=20000, worN=f0
            )

        h0 = np.array(h)
        gain = np.abs(h0)
        phase = np.angle(h, deg=True)

        return w, gain, phase

    def do_hw_normalization(self, input: np.ndarray, full_range=False) -> np.ndarray:
        """Normalization of the input to binary shifting, range [+1, -1]"""
        frame_out = np.zeros(shape=input.shape, dtype='float')
        offset = 0.5 if full_range else 0
        for ite, frame in enumerate(input):
            max_val = np.max(np.abs(frame))
            scale_val = 1 / np.ceil(np.log2(max_val))
            frame_out[ite, :] = offset + scale_val * frame

        return frame_out