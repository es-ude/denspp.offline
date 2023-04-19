import numpy as np
from scipy.signal import savgol_filter, find_peaks
from settings import Settings

class SDA:
    def __init__(self, setting: Settings):
        # --- Input
        self.sample_rate_adc = setting.fs_adc

        # --- Spike detection incl. thresholding and frame generation
        self.__dx_sda = setting.dx_sda
        self.__dx_neo = setting.dx_neo
        self.__dx_mteo = setting.dx_mteo

        self.__frame_mode = setting.mode_frame
        self.__mean_window = setting.x_window_mean

        self.__offset_frame_neg = round(setting.x_offset[0] * self.sample_rate_adc)
        self.__offset_frame_pos = round(setting.x_offset[1] * self.sample_rate_adc)
        self.offset_frame = self.__offset_frame_neg + self.__offset_frame_pos

        self.frame_length = round(setting.x_window_length * self.sample_rate_adc)
        self.frame_neg = round(setting.x_window_start * self.sample_rate_adc)
        self.frame_pos = self.frame_length - self.frame_neg

    def sda(self, xin: np.ndarray, mode: int) -> [np.ndarray, np.ndarray]:
        ksda = self.__dx_sda
        if ksda.size == 0:
            x_mteo = xin
        else:
            x_mteo = np.zeros(shape=(ksda.size, xin.size))
            for idx in range(0, ksda.size):
                ksda0 = int(ksda[idx])
                x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
                x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        x_sda = np.max(x_mteo, 0)
        x_thr = self.__thres(x_sda, mode)

        return x_sda, x_thr

    def sda_norm(self, xin: np.ndarray, mode: int) -> [np.ndarray, np.ndarray]:
        # Selection of SDA is made via the vector length of dXsda
        # applied on raw datastream
        x_sda = xin
        x_thr = self.__thres(x_sda, mode)

        return x_sda, x_thr

    def sda_neo(self, xin: np.ndarray, mode: int) -> [np.ndarray, np.ndarray]:
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        ksda0 = self.__dx_neo

        x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
        x_neo = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        x_sda = x_neo
        x_thr = self.__thres(x_sda, mode)

        return x_sda, x_thr

    def sda_mteo(self, xin: np.ndarray, mode: int) -> [np.ndarray, np.ndarray]:
        # performing mteo
        x_mteo = np.zeros(shape=(self.__dx_mteo.size, xin.size))

        for idx, ksda0 in enumerate(self.__dx_mteo):
            x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
            x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        x_sda = np.max(x_mteo, 0)
        x_thr = self.__thres(x_sda, mode)

        return x_sda, x_thr

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray) -> [np.ndarray, np.ndarray]:
        # Trigger generation
        result = ((xsda - xthr) > 0)
        xtrg = result.astype("int")

        if np.sum(xtrg) == 0:
            # Abort if no results are available
            pass
        else:
            # --- Extraction of x-positions
            mode = 1
            if mode == 0:
                # Findpeak
                width = 3
                x_pos = np.convolve(xtrg, np.ones(width), mode="same")
                (x_pos0, _) = find_peaks(x_pos, distance=self.frame_length)
            elif mode == 1:
                # Rising edge
                width = 3
                x_pos = np.convolve(xtrg, np.ones(width), mode="same")
                trigger_val = 0.5
                x_pos = np.flatnonzero((x_pos[:-1] < trigger_val) & (x_pos[1:] > trigger_val)) + 1

            x_pos0 = []
            for idx, val in enumerate(x_pos):
                if not idx == 0:
                    dx = val - x_pos[idx-1]
                    if dx >= self.frame_length:
                        x_pos0.append(val)

            # --- Generate frames
            lgth_frame = self.frame_length + self.offset_frame
            lgth_data = len(result)
            frame = []
            xpos_out = []
            for idx, pos_frame in enumerate(x_pos0):
                dx_neg = pos_frame - self.__offset_frame_neg
                dx_pos = pos_frame + lgth_frame
                if dx_neg > 0 and dx_pos < lgth_data:
                    frame.append(xraw[dx_neg:dx_pos])
                    xpos_out.append(pos_frame)

            frame = np.array(frame, dtype=int)
            xpos_out = np.array(xpos_out, dtype=int)

        return frame, xpos_out

    def frame_aligning(self, frame_in: np.ndarray, align_mode: int) -> np.ndarray:
        # ---Check if no results are available
        idx = 0
        frame_out = np.zeros(shape=(frame_in.shape[0], self.frame_length), dtype="int")
        for row in frame_in:
            frame0 = row
            frame = np.convolve(frame0, np.ones(2), mode="same")
            # --- Window of finding feature
            idx0 = int(self.frame_neg)
            idx1 = row.size - idx0

            # --- Aligning
            # TODO: Fenster-Methode einfügen
            search_frame = frame
            if align_mode == 0:     # no alignment
                max_pos = self.__offset_frame_neg + self.frame_neg
            elif align_mode == 1:   # align to maximum
                max_pos = np.argmax(search_frame, axis=None)
            elif align_mode == 2:   # align to minimum
                max_pos = np.argmin(search_frame, axis=None)
            elif align_mode == 3:   # align to positive turning point
                max_pos = np.argmax(np.diff(search_frame), axis=None)
                max_pos = max_pos + 1
            elif align_mode == 4:   # align to negative turning point
                max_pos = np.argmin(np.diff(search_frame), axis=None)
                max_pos = max_pos + 1

            # --- Do the Aligning (Detection of non-full frames due to offset edges)
            dxneg = max_pos - self.frame_neg
            dxpos = dxneg + self.frame_length

            if dxpos > len(frame0):
                # Missing informations at upper edge
                mat = np.ones(shape=(1, np.abs(dxpos - len(frame0) +1))) * frame0[-1]
                frame1 = np.concatenate((frame0[dxneg:-1], mat), axis=None)
            elif dxneg <= 0:
                # Missing informations at downer edge
                mat = np.ones(shape=(1, np.abs(dxneg))) * frame0[0]
                frame1 = np.concatenate((mat, frame0[0:dxpos]), axis=None)
            else: # Normal state
                frame1 = frame0[dxneg:dxpos]

            frame_out[idx,:] = frame1
            idx += 1

        return frame_out

    def __thres(self, xin: np.ndarray, mode: int) -> np.ndarray:
        if mode == 1:    # standard derivation of background activity
            x_out = 0 * xin + 8 / 0.6745 * np.mean(np.abs(xin))
        elif mode == 2:  # Automated calculation of threshold (use by BlackRock)
            x_out = 0 * xin + 4.5 * np.sqrt(np.sum(xin**2 / len(xin)))
        elif mode == 3:  # Mean value
            x_out = 10 * self.__movmean(xin, self.__mean_window)
        elif mode == 4:  # Lossy Peak detection
            x_out = 10 * self.__movmean(np.abs(xin), self.__mean_window)
        elif mode == 5:  # Window Mean method for Max-detection
            x_out = 0 * xin
            gain = 10
            window_length = 20
            window_mean = 200
            for i in range(np.floor(xin.size / window_length).astype("int")):
                # TODO: Problem finden und lösen
                x0 = np.array([[1, window_length]]) + (i - 1) * window_length
                x_out[x0[0, 0] : x0[0, 1]] = np.max(xin[x0[0, 0] : x0[0, 1]])

            x_out = gain * self.__movmean(x_out, window_mean)
        elif mode == 6: # Salvan-Goley-Fiter
            if xin.dtype == np.ushort:
                x0 = xin.astype(np.double)
            else:
                x0 = xin
            x_out = savgol_filter(x0, 3, 31)

        x_out = np.floor(x_out)

        return x_out

    def __movmean(self, xin: np.ndarray, window: int) -> np.ndarray:
        xout = np.convolve(xin, np.ones(window)/window, mode='same')
        return xout

    def __hl_envelopes_idx(self, signal: np.ndarray, dmin=1, dmax=1, split=False):
        """
        Input :
        s: 1d-array, data signal from which to extract high and low envelopes
        dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
        split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
        Output :
        lmin,lmax : high/low envelope idx of input signal s
        """

        # locals min
        lmin = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1
        # locals max
        lmax = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1

        if split:
            # s_mid is zero if s centered around x-axis or more generally mean of signal
            s_mid = np.mean(signal)
            # pre-sorting of locals min based on relative position with respect to s_mid
            lmin = lmin[signal[lmin] < s_mid]
            # pre-sorting of local max based on relative position with respect to s_mid
            lmax = lmax[signal[lmax] > s_mid]

        # global max of dmax-chunks of locals max
        lmin = lmin[[i + np.argmin(signal[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]]
        # global min of dmin-chunks of locals min
        lmax = lmax[[i + np.argmax(signal[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]]

        return lmin, lmax
