import numpy as np
from dataclasses import dataclass
from scipy.signal import savgol_filter, find_peaks, iirfilter, lfilter
from denspp.offline.data_process.transformation import transformation_window_method


@dataclass
class SettingsSDA:
    """Configuration class for defining the Spike Detection Algorithm (SDA)
    Attributes:
        fs:             Sampling rate [Hz]
        dx_sda:         Position difference for extracting SDA method. Configuration with length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        mode_align:     Aligning mode of the detected spike frames [0: None, 1: Max, 2: Min, 3: Positive turning point, 4: Negative turning point, 5: Absolute maximum]
        t_frame_lgth:   Floating value with total window length [s]
        t_frame_start:  Floating value with time point for aligned position [s]
        dt_offset:      Time offset for the first larger spike window [neg, pos]
        t_dly:          Applied time delay between spike buffer and SDA method
        window_size:    Integer value of the window for smoothing the SDA output
        thr_gain:       Floating value with amplification factor on SDA output
        thr_min_value:  Integer value with minimum threshold value on SDA output

    """
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

    @property
    def get_integer_for_negative_offset(self) -> int:
        """Getting the integer offset for negative offset in building the spike window"""
        return round(self.dt_offset[0] * self.fs)

    @property
    def get_integer_for_positive_offset(self) -> int:
        """Getting the integer offset for positive offset in building the spike window"""
        return round(self.dt_offset[1] * self.fs)

    @property
    def get_integer_offset_total(self) -> int:
        """Getting the total integer offset in building the spike window"""
        return self.get_integer_for_negative_offset + self.get_integer_for_positive_offset

    @property
    def get_integer_spike_frame(self) -> int:
        """Getting the integer for total length of a spike window"""
        return round(self.t_frame_lgth * self.fs)

    @property
    def get_integer_spike_start(self) -> int:
        """Getting the integer for starting the aligned method on each spike window"""
        return round(self.t_frame_start * self.fs)


RecommendedSettingsSDA = SettingsSDA(
    fs=20e3,
    dx_sda=[1],
    mode_align=1,
    t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
    dt_offset=[0.1e-3, 0.1e-3],
    t_dly=0.3e-3,
    window_size=7,
    thr_gain=1.0,
    thr_min_value=100.0
)


class SpikeDetection:
    def __init__(self, setting: SettingsSDA):
        self.settings = setting

        # --- Parameters for Frame generation and aligning
        self.__offset_frame_neg = self.settings.get_integer_for_negative_offset
        self.frame_length = self.settings.get_integer_spike_frame
        self.frame_length_total = self.frame_length + self.settings.get_integer_offset_total
        self.frame_start = self.settings.get_integer_spike_start
        self.frame_ends = self.frame_length - self.frame_start

    # --------- Pre-Processing of SDA -------------
    def time_delay(self, uin: np.ndarray) -> np.ndarray:
        """Applying a time delay on the input signal
        :param uin:     Numpy array with transient input signal / data stream
        :return:        Numpy array with time-delayed signal
        """
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

    # --------- Spike Detection Algorithm -------------
    def sda_norm(self, xin: np.ndarray) -> np.ndarray:
        """Normal spike detection algorithm"""
        return xin

    def sda_neo(self, xin: np.ndarray) -> np.ndarray:
        """Applying Non-Linear Energy Operator (NEO, same like Teager-Kaiser-Operator) with dx_sda = 1 or kNEO with dx_sda > 1"""
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

    def sda_eed(self, xin: np.ndarray, f_hp: float=150.0) -> np.ndarray:
        """Applying the enhanced energy-derivation operator (eED) on input signal
        :param xin:     Numpy array with transient neural signal with spikes
        :param f_hp:    Floating value for applied high-pass filter in this function
        """
        filter = iirfilter(
            N=2, Wn=2 * f_hp / self.settings.fs, ftype="butter", btype="highpass",
            analog=True, output='ba'
        )
        eed = np.array(lfilter(filter[0], filter[1], xin))
        return np.square(eed)

    def sda_spb(self, xin: np.ndarray, f_bp: list=(100.0, 1000.0)) -> [np.ndarray, np.ndarray]:
        """Performing the spike detection with spike band-power estimation [Nason et al., 2020]
        :param xin:     Numpy array with transient neural signal with spikes
        :param f_bp:    List with floating value for applied band-power filtering in this function
        """
        filter = iirfilter(N=2, Wn=2 * np.array(f_bp) / self.settings.fs, ftype="butter", btype="bandpass", analog=False, output='ba')
        filt0 = lfilter(filter[0], filter[1], xin)
        sbp = self.smoothing_1d(np.abs(filt0), int(1e-3 * self.settings.fs), 'gaussian')
        return np.floor(sbp)

    def sda_smooth(self, xin: np.ndarray, method: str='') -> np.ndarray:
        """Smoothing the input with defined window ['', 'hamming', 'gaussian', 'bartlett', 'blackman']"""
        return self.smoothing_1d(xin, 4 * self.settings.dx_sda[0] + 1, method)

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
        """Getting frame position of SDA output and thresholding
        :param xsda:    Numpy array with transient signal of SDA output
        :param xthr:    Numpy array with applied thresholding
        :return:        Numpy array with positions where a spike frame is available
        """
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

    def __frame_extraction(self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int=0) -> [list, list, list, list]:
        """Function for extracting the frames from transient data steam
       :param xraw:         Numpy array with transient raw input signal
       :param xpos:         Numpy array with positions where a spike frame is available
       :param xoffset:      Integer value with offset to generate larger spike windows
       :return:             Tuple with [0] first generated spike frame (large), [1] original position, [2] aligned spike frames [3] and position
        """
        f0 = self.__offset_frame_neg
        f1 = f0 + int(self.frame_length_total / 2)

        orig_frames = list()
        alig_frames = list()
        orig_xpos = list()
        alig_xpos = list()
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
            x_mid = x_neg0 + self.frame_start
            # Abort condition if values are out of range
            if x_neg1 < 0 or x_pos1 > xraw.size:
                continue
            frame1 = xraw[x_neg1:x_pos1]
            # --- Add to output
            orig_frames.append(frame0)
            orig_xpos.append(pos)
            alig_frames.append(frame1)
            alig_xpos.append(x_mid)

        return orig_frames, orig_xpos, alig_frames, alig_xpos

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray) -> [list, list]:
        """Frame generation of SDA output and threshold
        :param xraw:    Numpy array with transient raw data
        :param xsda:    Numpy array with transient signal from spike detection algorithm
        :param xthr:    Numpy array with transient signal with thresholding
        :return:        Tuple with [0] large frame (non-aligned) and [1] aligned frame
        """
        xpos = self.frame_position(xsda, xthr)
        frames_orig, frames_xpos_orig, frames_align, frames_xpos_align = self.__frame_extraction(xraw, xpos)

        frames_orig = np.array(frames_orig, dtype=np.dtype('int16'))
        frames_xpos_orig = np.array(frames_xpos_orig, dtype=np.dtype('uint64'))
        frames_align = np.array(frames_align, dtype=np.dtype('int16'))
        frames_xpos_align = np.array(frames_xpos_align, dtype=np.dtype('uint64'))

        frames_out_orig = [frames_orig, frames_xpos_orig, np.zeros(shape=frames_xpos_orig.shape, dtype=np.dtype('uint8'))]
        frames_out_align = [frames_align, frames_xpos_align, np.zeros(shape=frames_xpos_align.size, dtype=np.dtype('uint8'))]

        return frames_out_orig, frames_out_align

    def frame_generation_pos(self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Frame generation from already detected positions (in datasets with groundtruth)
        :param xraw:    Numpy array with transient raw data
        :param xpos:    Numpy array with position where a spike frame is available
        :param xoffset: Integer value with offset to generate larger spike windows
        :return:        Tuple with [0] original (large) spike frame, [1] algined spike frame and [2] positions

        """
        frames_orig, frames_xpos, frames_algn,_ = self.__frame_extraction(
            xraw=xraw,
            xpos=xpos,
            xoffset=xoffset
        )
        frames_orig = np.array(frames_orig, dtype=np.dtype('int16'))
        frames_algn = np.array(frames_algn, dtype=np.dtype('int16'))
        frames_xpos = np.array(frames_xpos, dtype=np.dtype('uint64'))

        return frames_orig, frames_algn, frames_xpos

    # --------- Frame Aligning -------------
    def __frame_correction(self, frame_in: np.ndarray, dx_neg: int, dx_pos: int) -> np.ndarray:
        """Do a frame correction if set x-values are out-of-range
        :param frame_in:    Numpy array with detected spike frames (orignial, large windows)
        :param dx_neg:      Integer value with negative x-values to start reducing
        :param dx_pos:      Integer value with positive x-values to stop reducing
        :return:            Numpy array with corrected spike frames (in final size)
        """
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
        """Extracting aligning position of spike frames
        :param frame_in:    Numpy array with detected spike frames
        :return:            List with [0] negative position and [1] positive position for aligning spike frame
        """
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
        """Aligning method for detected spike frames
        :param frame_in:    Numpy array with detected spike frames after detection
        :return:            Numpy array with aligned spike frames (final)
        """
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

    @staticmethod
    def smoothing_1d(xin: np.ndarray, window_size: int, method: str='hamming') -> np.ndarray:
        """Smoothing the input
        :param xin:         Numpy array with transient signal of SDA output
        :param window_size: Integer value for applied smoothing method
        :param method:      String with method for smoothing input
        :return:            Numpy array with smoothed SDA output
        """
        window = transformation_window_method(window_size, method)
        return np.convolve(xin, window, mode='same') # / np.sum(window), mode='same')
