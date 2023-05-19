import dataclasses
import numpy as np
from scipy.signal import savgol_filter, find_peaks, iirfilter, lfilter

@dataclasses.dataclass
class SettingsSDA:
    fs: float
    dx_sda: list
    mode_align: int
    t_frame_lgth: float
    t_frame_start: float
    dt_offset: list
    t_dly: float

class RecommendedSettingsSDA(SettingsSDA):
    def __init__(self):
        super().__init__(
            fs=20e3,
            dx_sda=[1],
            mode_align=1,
            t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
            dt_offset=[0.4e-3, 0.3e-3],
            t_dly=0.3e-3
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

    def thres_sd(self, xin: np.ndarray) -> np.ndarray:
        """Apply standard derivation as threshold"""
        return  np.ones(xin.size) + 8 * np.mean(np.abs(xin))

    def thres_blackrock(self, xin: np.ndarray) -> np.ndarray:
        """Automated calculation of threshold (use by BlackRock)"""
        return np.ones(xin.size) + 4.5 * np.sqrt(np.sum(xin ** 2 / len(xin)))

    def thres_blackrock_runtime(self, xin: np.ndarray) -> np.ndarray:
        """Runtime calculation of threshold (use by BlackRock)"""
        x_thr = np.zeros(shape=xin.shape, dtype=int)
        window = 50
        mean = 0
        for idx, val in enumerate(xin):
            if idx < window:
                xin0 = xin[0:idx]
                N0 = idx + 1
            else:
                xin0 = xin[idx-window:idx]
                N0 = window
            x_thr[idx] = np.sqrt(np.sum((xin0 - mean) ** 2) / N0)
            x_thr[0:window] = x_thr[-window-1:-1]
        return 5 * x_thr

    def thres_movmean(self, xin: np.ndarray) -> np.ndarray:
        """Applying moving average filter on input"""
        width = 100
        return 10 * np.convolve(xin, np.ones(width)/width, mode='same')

    def thres_salvan_golay(self, xin: np.ndarray) -> np.ndarray:
        """Applying a Salvan-Golay Filter on input signal"""
        return savgol_filter(xin, self.frame_length, 3)

    # --------- Spike Detection Algorithm -------------
    def sda_norm(self, xin: np.ndarray) -> np.ndarray:
        """Normal spike detection algorithm (Durchschleifen)"""
        return xin

    def sda_eed(self, xin: np.ndarray, fs: float) -> np.ndarray:
        """Applying the enhanced energy-derivation operator (eED) on input signal"""
        fhp = 150
        filter = iirfilter(
            N=2, Wn=2 * fhp / fs, ftype="butter", btype="highpass",
            analog=True, output='ba'
        )
        eed = np.array(lfilter(filter[0], filter[1], xin))
        return np.square(eed)

    def sda_neo(self, xin: np.ndarray) -> np.ndarray:
        """Applying Non-Linear Energy Operator (NEO, same like Teager-Kaiser-Operator) with dx_sda = 1 or kNEO with dx_sda > 1"""
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        ksda0 = self.settings.dx_sda[0]
        x0 = np.floor(xin[ksda0:-ksda0] ** 2 - xin[:-2 * ksda0] * xin[2 * ksda0:])
        x_sda = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)
        return x_sda

    def sda_mteo(self, xin: np.ndarray) -> np.ndarray:
        """Applying Multiresolution Teager Energy Operator (MTEO) on input signal"""
        x_mteo = np.zeros(shape=(len(self.settings.dx_sda), xin.size))
        for idx, ksda0 in enumerate(self.settings.dx_sda):
            x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
            x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        return np.max(np.floor(x_mteo), 0)

    def sda_snn(self, xin: np.ndarray, y_thr: float) -> [np.ndarray, np.ndarray]:
        """Applying the spiking neural network (SNN) converter in order to extract spike pattern"""
        y_snn = np.zeros(shape=xin.size)
        y_int = np.zeros(shape=xin.size)
        y_int0 = 0.0
        gain = 1.0
        for idx, val in enumerate(np.abs(xin)):
            if y_int0 >= y_thr:
                y_int0 = 0.0
                y_snn[idx] = 1
            else:
                y_int0 += gain * val
                y_snn[idx] = 0

            y_int[idx] = y_int0

        return y_snn, y_int

    # --------- Frame Generation -------------
    def __gen_findpeaks(self, xtrg: np.ndarray, width: int) -> list:
        """Method for x_pos determination with findpeaks"""
        x_pos = np.convolve(xtrg, np.ones(width) / width, mode="same")
        (x_pos0, _) = find_peaks(x_pos, distance=self.frame_length)
        return list(x_pos0)

    def __gen_risingedge(self, xtrg: np.ndarray, width: int) -> list:
        """Method for x_pos determination with looking for rising edge"""
        trigger_val = 0.5
        x_pos = np.convolve(xtrg, np.ones(width) / width, mode="same")
        x_pos = np.argwhere(x_pos >= trigger_val)
        x_pos0 = [x_pos[0]]
        for idx, pos in enumerate(x_pos):
            dx = pos - x_pos0[-1]
            if dx >= self.frame_length:
                x_pos0.append(pos)
        return x_pos0

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Frame generation of SDA output and threshold"""
        xtrg = ((xsda - xthr) > 0).astype("int")
        # --- Extraction of x-positions
        mode = 1
        width = 1
        if mode == 0:
            xpos = self.__gen_findpeaks(xtrg, width)
        else:
            xpos = self.__gen_risingedge(xtrg, width)

        # --- Generate frames
        frames_orig = []
        frames_align = []
        xpos_out = []

        f0 = self.__offset_frame_neg
        f1 = f0 + int(self.frame_length_total / 2)

        for idx, pos_frame in enumerate(xpos):
            # --- Original larger frame
            x_neg0 = int(pos_frame - self.__offset_frame_neg)
            x_pos0 = int(pos_frame + self.frame_length_total)
            frame0 = xraw[x_neg0:x_pos0]
            # --- Aligned frame
            x_neg1 = int(x_neg0 + f0 + self.get_aligning_position(frame0[f0:f1])[0])
            x_pos1 = int(x_neg1 + self.frame_length)
            frame1 = xraw[x_neg1:x_pos1]
            # --- Add to output
            frames_orig.append(frame0)
            frames_align.append(frame1)
            xpos_out.append(pos_frame)

        frames_orig = np.array(frames_orig, dtype=int)
        frames_align = np.array(frames_align, dtype=int)
        xpos_out = np.array(xpos_out, dtype=int)

        return frames_orig, frames_align, xpos_out

    # --------- Frame Aligning -------------
    def __frame_correction(self, frame_in: np.ndarray, dx_neg: int, dx_pos: int) -> np.ndarray:
        raise NotImplementedError
        # TODO: Methode noch implementieren

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

    def get_aligning_position(self, frame_in: np.ndarray) -> [int, int]:
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
            frame_out[idx, :] = frame0[f0+dxneg:dxpos]

        return frame_out
