import dataclasses
import numpy as np
from scipy.signal import savgol_filter, find_peaks, iirfilter, lfilter
from numpy import hamming, bartlett, kaiser


@dataclasses.dataclass
class SettingsSDA:
    fs: float
    dx_sda: list
    mode_align: int
    t_frame_lgth: float
    t_frame_start: float
    dt_offset: list
    t_dly: float
    window_size: int
    thr_gain: float
    thr_min_value: float


class RecommendedSettingsSDA(SettingsSDA):
    def __init__(self):
        super().__init__(
            fs=20e3,
            dx_sda=[1],
            mode_align=1,
            t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
            dt_offset=[0.1e-3, 0.1e-3],
            t_dly=0.3e-3,
            window_size=7,
            thr_gain=1.0,
            thr_value=100.0
        )


class SpikeDetection:
    def __init__(self, setting: SettingsSDA):
        self.settings = setting

        # --- Parameters for Frame generation and aligning
        self.__offset_frame_neg = round(self.settings.dt_offset[0] * self.settings.fs)
        self.__offset_frame_pos = round(self.settings.dt_offset[1] * self.settings.fs)
        self.offset_frame = self.__offset_frame_neg + self.__offset_frame_pos

        self.frame_length = round(self.settings.t_frame_lgth * self.settings.fs)
        self.frame_length_total = self.frame_length + self.offset_frame
        self.frame_start = round(self.settings.t_frame_start * self.settings.fs)
        self.frame_ends = self.frame_length - self.frame_start

    # --------- Pre-Processing of SDA -------------
    def time_delay(self, uin: np.ndarray) -> np.ndarray:
        """Applying a time delay on the input signal"""
        set_delay = round(self.settings.t_dly * self.settings.fs)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
        return uout

    def thres_const(self, xin: np.ndarray) -> np.ndarray:
        """Applying a constant value for thresholding"""
        return np.zeros(shape=xin.size) + self.settings.thr_min_value

    def thres_mad(self, xin: np.ndarray) -> np.ndarray:
        """Apply the median absolute derivation (MAD) as threshold"""
        C = self.settings.thr_gain
        return np.zeros(shape=xin.size) + C * np.median(np.abs(xin - np.mean(xin))/0.6745)

    def thres_ma(self, xin: np.ndarray) -> np.ndarray:
        """Applying the mean absolute (Moving average) as threshold"""
        C = self.settings.thr_gain
        M = self.settings.window_size
        return C * 1.25 * np.convolve(np.abs(xin), np.ones(M)/M, mode='same')

    def thres_rms(self, xin: np.ndarray) -> np.ndarray:
        """Applying the root-mean-squre (RMS) on neural input"""
        C = self.settings.thr_gain
        M = self.settings.window_size
        return C * np.sqrt(np.convolve(xin ** 2, np.ones(M) / M, mode='same'))

    def thres_winsorization(self, xin: np.ndarray) -> np.ndarray:
        """Applying the winsorization method on input"""
        C = self.settings.thr_gain
        M = self.settings.window_size
        noise1 = self.thres_ma(xin)
        noise2 = np.zeros(shape=xin.shape)
        for idx, val in enumerate(np.abs(xin)):
            if val < noise1[idx]:
                noise2[idx] = val
            else:
                noise2[idx] = noise1[idx]

        return C * 1.58 * np.convolve(noise2, np.ones(M)/M, 'valid')

    def thres_blackrock(self, xin: np.ndarray) -> np.ndarray:
        """Automated rms calculation of threshold (use by BlackRock)"""
        C = self.settings.thr_gain
        return np.zeros(shape=xin.size) + C * 4.5 * np.sqrt(np.sum(xin ** 2 / len(xin)))

    def thres_blackrock_runtime(self, xin: np.ndarray) -> np.ndarray:
        """Runtime std calculation of threshold (use by BlackRock)"""
        x_thr = np.zeros(shape=xin.shape, dtype=int)
        C = self.settings.thr_gain
        M = self.settings.window_size
        mean = 0
        for idx, val in enumerate(xin):
            if idx < M:
                N0 = idx + 1
            else:
                N0 = M

            xin0 = xin[idx-M:idx]
            x_thr[idx] = np.sqrt(np.sum((xin0 - mean) ** 2) / N0)
            x_thr[0:M] = x_thr[-M-1:-1]
        return C * x_thr

    def thres_salvan_golay(self, xin: np.ndarray) -> np.ndarray:
        """Applying a Salvan-Golay Filter on input signal"""
        C = self.settings.thr_gain
        return C * savgol_filter(xin, self.frame_length, 3)

    #TODO: Methode aus Paper 10.1088/1741-2552/accece implementieren
    def thres_firing_rate(self, xin: np.ndarray) -> np.ndarray:
        """Applying the firing-rate decoding of the threshold methode"""
        return NotImplementedError

    # --------- Spike Detection Algorithm -------------
    def sda_norm(self, xin: np.ndarray) -> np.ndarray:
        """Normal spike detection algorithm"""
        return xin

    def sda_neo(self, xin: np.ndarray) -> np.ndarray:
        """Applying Non-Linear Energy Operator (NEO, same like Teager-Kaiser-Operator) with dx_sda = 1 or kNEO with dx_sda > 1"""
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        ksda0 = self.settings.dx_sda[0]
        x_neo0 = np.floor(xin[ksda0:-ksda0] ** 2 - xin[:-2 * ksda0] * xin[2 * ksda0:])
        x_neo = np.concatenate([x_neo0[:ksda0, ], x_neo0, x_neo0[-ksda0:, ]], axis=None)
        return x_neo

    def sda_mteo(self, xin: np.ndarray) -> np.ndarray:
        """Applying Multiresolution Teager Energy Operator (MTEO) on input signal"""
        x_mteo = np.zeros(shape=(len(self.settings.dx_sda), xin.size))
        for idx, ksda0 in enumerate(self.settings.dx_sda):
            x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
            x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        return np.max(np.floor(x_mteo), 0)

    def sda_ado(self, xin: np.ndarray) -> np.ndarray:
        """Applying the absolute difference operator (ADO) on input signal"""
        ksda0 = self.settings.dx_sda[0]
        x_sda = np.floor(np.absolute(xin[ksda0:, ] - xin[:-ksda0, ]))
        x_ado = np.concatenate([x_sda[:ksda0], x_sda], axis=None)
        return x_ado

    def sda_aso(self, xin: np.ndarray) -> np.ndarray:
        """Applying the amplitude slope operator (ASO, k for window size) on input signal"""
        ksda0 = self.settings.dx_sda[0]
        x_sda = np.floor(xin[ksda0:, ] * (xin[ksda0:, ] - xin[:-ksda0, ]))
        x_aso = np.concatenate([x_sda[:ksda0], x_sda], axis=None)
        return x_aso

    def sda_eed(self, xin: np.ndarray, fs: float) -> np.ndarray:
        """Applying the enhanced energy-derivation operator (eED) on input signal"""
        fhp = 150
        filter = iirfilter(
            N=2, Wn=2 * fhp / fs, ftype="butter", btype="highpass",
            analog=True, output='ba'
        )
        eed = np.array(lfilter(filter[0], filter[1], xin))
        return np.square(eed)

    # TODO: Implementierung SBP Funktion
    def sda_spb(self, xin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performing the spike detection with spike band-power estimation [Nason et al., 2020]"""
        return NotImplementedError

    def sda_smooth(self, xin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Smoothing the input"""
        window = hamming(4 * self.settings.dx_sda[0] + 1)
        gain_window = np.sum(window)
        gain = 1
        xout = np.convolve(xin, gain * window/gain_window, mode='same')
        return xout, window

    # --------- Frame Generation -------------
    def __gen_findpeaks(self, xtrg: np.ndarray, width: int) -> list:
        """Method for x_pos determination with findpeaks"""
        x_pos = np.convolve(xtrg, np.ones(width) / width, mode="same")
        (x_pos0, _) = find_peaks(x_pos, distance=self.frame_length)
        return list(x_pos0)

    def __gen_risingedge(self, xtrg: np.ndarray, width: int) -> list:
        """Method for x_pos determination with looking for rising edge"""
        trigger_val = 0.5
        x_pos0 = np.convolve(xtrg, np.ones(width) / width, mode="same")
        x_pos1 = np.argwhere(x_pos0 >= trigger_val)
        x_out = list()
        x_out.append(x_pos1[0][0])
        for idx, pos in enumerate(x_pos1):
            dx = pos[0] - x_out[-1]
            if dx >= self.frame_length:
                x_out.append(pos[0])
        return x_out

    def frame_position(self, xsda: np.ndarray, xthr: np.ndarray) -> np.ndarray:
        """Getting frame position of SDA output and thresholding"""
        xtrg = ((xsda - xthr) > 0).astype("int")
        # --- Extraction of x-positions
        mode = 0
        width = 1
        if mode == 0:
            xpos = self.__gen_findpeaks(xtrg, width)
        else:
            xpos = self.__gen_risingedge(xtrg, width)

        xpos_out = np.array(xpos, dtype=int)

        return xpos_out

    def __frame_extraction(self, xraw: np.ndarray, xpos: np.ndarray, xoffset=0) -> [list, list]:
        """Extraction of the frames"""
        f0 = self.__offset_frame_neg
        f1 = f0 + int(self.frame_length_total / 2)

        frames_orig = []
        frames_align = []
        for idx, pos in enumerate(xpos):
            # --- Original larger frame
            x_neg0 = int(pos - self.__offset_frame_neg + xoffset)
            x_pos0 = int(x_neg0 + self.frame_length_total)
            # Abort condition if values are out of range
            if x_neg0 < 0 or x_pos0 > xraw.size:
                continue
            frame0 = xraw[x_neg0:x_pos0]
            # --- Aligned frame
            x_neg1 = x_neg0 + f0 + self.get_aligning_position(frame0[f0:f1])[0]
            x_pos1 = x_neg1 + self.frame_length
            # Abort condition if values are out of range
            if x_neg1 < 0 or x_pos1 > xraw.size:
                continue
            frame1 = xraw[x_neg1:x_pos1]
            # --- Add to output
            frames_orig.append(frame0)
            frames_align.append(frame1)

        return frames_orig, frames_align

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray) -> [list, list]:
        """Frame generation of SDA output and threshold"""
        xpos = self.frame_position(xsda, xthr)
        xpos_aligned = 0 * xpos

        frames_orig, frames_align = self.__frame_extraction(xraw, xpos)

        frames_orig = np.array(frames_orig, dtype=np.dtype('int16'))
        frames_align = np.array(frames_align, dtype=np.dtype('int16'))
        frames_out0 = [frames_orig, xpos, np.zeros(shape=(xpos.size,), dtype=np.dtype('uint32'))]
        frames_out1 = [frames_align, xpos_aligned, np.zeros(shape=(xpos_aligned.size, ), dtype=int)]

        return frames_out0, frames_out1

    def frame_generation_pos(self, xraw:np.ndarray, xpos: np.ndarray, xoffset: int) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Frame generation from already detected positions (in datasets with groundtruth)"""
        frames_orig, frames_align = self.__frame_extraction(xraw, xpos, xoffset=xoffset)

        frames_orig = np.array(frames_orig, dtype=np.dtype('int16'))
        frames_align = np.array(frames_align, dtype=np.dtype('int16'))
        xpos_out = np.array(xpos, dtype=np.dtype('uint32'))

        return frames_orig, frames_align, xpos_out

    # --------- Frame Aligning -------------
    def __frame_correction(self, frame_in: np.ndarray, dx_neg: int, dx_pos: int) -> np.ndarray:
        """Do a frame correction if set x-values are out-of-range"""
        if dx_pos > frame_in.size:  # Add right side
            mat = np.ones(shape=(1, np.abs(dx_pos - len(frame_in) + 1))) * frame_in[-1]
            frame1 = np.concatenate((frame_in[dx_neg:-1], mat), axis=None)
        elif dx_neg <= 0:       # Add left side
            mat = np.ones(shape=(1, np.abs(dx_neg))) * frame_in[0]
            frame1 = np.concatenate((mat, frame_in[0:dx_pos]), axis=None)
        else:                   # Normal state
            frame1 = frame_in[dx_neg:dx_pos]
        return frame1

    def __frame_align_none(self, frame_in: np.ndarray) -> [int, int]:
        """None-aligning the detected spike frames (only smaller window)"""
        x_neg = self.__offset_frame_neg
        x_pos = x_neg + self.frame_length
        return x_neg, x_pos

    def __frame_align_max(self, frame_in: np.ndarray) -> [int, int]:
        """Aligning the detected spike frames to maximum"""
        x_start = np.argmax(frame_in, axis=None)
        dxneg = x_start - self.frame_start
        dxpos = dxneg + self.frame_length
        return dxneg, dxpos

    def __frame_align_min(self, frame_in: np.ndarray) -> [int, int]:
        """Aligning the detected spike frames to minimum"""
        x_start = np.argmin(frame_in, axis=None)
        dxneg = x_start - self.frame_start
        dxpos = dxneg + self.frame_length
        return dxneg, dxpos

    def __frame_align_ptp(self, frame_in: np.ndarray) -> [int, int]:
        """Aligning the detected spike frames to positive turning point"""
        max_pos = 1 + np.argmax(np.diff(frame_in), axis=None)
        dxneg = max_pos - self.frame_start
        dxpos = dxneg + self.frame_length
        return dxneg, dxpos

    def __frame_align_ntp(self, frame_in: np.ndarray) -> [int, int]:
        """Aligning the detected spike frames to negative turning point"""
        max_pos = 1 + np.argmin(np.diff(frame_in), axis=None)
        dxneg = max_pos - self.frame_start
        dxpos = dxneg + self.frame_length
        return dxneg, dxpos

    def __frame_first_absmax(self, frame_in: np.ndarray) -> [int, int]:
        """Aligning the detected spike frames to first min/max peak"""
        x_max = np.argmax(frame_in, axis=None)
        x_min = np.argmin(frame_in, axis=None)
        x_start = np.min([x_max, x_min])
        dxneg = x_start - self.frame_start
        dxpos = dxneg + self.frame_length

        return dxneg, dxpos

    def get_aligning_position(self, frame_in: np.ndarray) -> [int, int]:
        """Extracting aligning position of spike frames"""
        align_mode = self.settings.mode_align
        if align_mode == 0:
            dxneg, dxpos = self.__frame_align_none(frame_in)
        elif align_mode == 1:
            dxneg, dxpos = self.__frame_align_max(frame_in)
        elif align_mode == 2:
            dxneg, dxpos = self.__frame_align_min(frame_in)
        elif align_mode == 3:
            dxneg, dxpos = self.__frame_align_ptp(frame_in)
        elif align_mode == 4:
            dxneg, dxpos = self.__frame_align_ntp(frame_in)
        elif align_mode == 5:
            dxneg, dxpos = self.__frame_first_absmax(frame_in)
        else:
            dxneg, dxpos = [0, 0]

        return dxneg, dxpos

    def do_aligning_frames(self, frame_in: np.ndarray) -> np.ndarray:
        """Aligning method for detected spike frames"""
        frame_out = np.zeros(shape=(frame_in.shape[0], self.frame_length), dtype="int")
        # --- Window method
        f0 = self.__offset_frame_neg
        f1 = f0 + int(frame_in.shape[1] / 2)
        # --- Aligning
        for idx, frame0 in enumerate(frame_in):
            frame_sel = frame0[f0:f1]
            dxneg, dxpos = self.get_aligning_position(frame_sel)
            frame_out[idx, :] = self.__frame_correction(frame0, dxneg, dxpos)

        return frame_out
