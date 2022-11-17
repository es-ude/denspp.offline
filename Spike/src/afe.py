import os
from fractions import Fraction

import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly, savgol_filter, find_peaks

from settings import Settings


class AFE:
    def __init__(self, setting: Settings):
        self.realtime_mode = setting.realtime_mode
        self.udd = setting.udd
        self.uss = setting.uss
        self.ucm = (self.udd + self.uss) / 2
        # TODO: Obere Liste entrümpeln

        #  --- Variables for pre-amplification
        self.a_iir_lpf = None
        self.b_iir_lpf = None
        self.mem_iir_ana = None
        # --- Variables for ADC
        self.n_bit_adc = 12
        self.lsb = None
        self.partition_adc = None
        self.sample_rate_adc = None
        self.sample_rate_ana = None
        self.p_ratio = None
        self.q_ratio = None
        self.oversampling_ratio = None
        self.oversampling_cnt = None
        self.x_old_adc = None
        self.u_old_adc = None
        # --- Variables for digital filtering
        self.n_filt_dig = None
        self.f_range_dig = None
        self.a_iir_dig = None
        self.b_iir_dig = None
        self.mem_iir_dig = None
        self.x_old_dig = None
        # --- Variables for Spike detection
        self.thr_min = 1e-8
        self.dx_sda = 1
        self.mode_sda = 0
        self.mem_sda = None
        self.mem_thres = None
        self.x_sda_old = None
        # --- Variables for Frame Generation and Aligning
        self.en_frame = 0
        self.mem_frame = None
        self.input_delay = None
        self.x_delta_neg = None
        self.x_delta_pos = None
        self.x_offset = None
        self.x_window_length = None
        # --- Properties for Labeling datasets
        self.no_cluster = 5

        self.v_pri = setting.gain_pre
        self.n_filt_ana = setting.n_filt_ana
        self.f_filt_ana = setting.f_filt_ana
        iir_ana_result = butter(self.n_filt_ana, 2 * self.f_filt_ana[0, :] / setting.desired_fs, "bandpass")
        (self.b_iir_ana, self.a_iir_ana) = iir_ana_result[0], iir_ana_result[1]
        self.mem_iir_ana = self.ucm + np.zeros((1, len(self.b_iir_ana) - 1))
        self.input_delay = setting.input_delay

        self.u_range = self.ucm + np.array([-1, 1]) * setting.d_uref
        self.n_bit_adc = setting.n_bit_adc
        self.lsb = np.diff(self.u_range) / np.power(2, self.n_bit_adc)
        self.partition_adc = np.arange(self.u_range[0] + self.lsb / 2, self.u_range[1] + self.lsb / 2, self.lsb)
        self.oversampling_ratio = setting.oversampling
        self.oversampling_cnt = 1
        self.sample_rate_adc = setting.sample_rate
        self.sample_rate_ana = setting.desired_fs
        # TODO: Funktion funktioniert nicht
        (self.p_ratio, self.q_ratio) = (
            Fraction(self.sample_rate_adc * self.oversampling_ratio / self.sample_rate_ana)
            .limit_denominator(100)
            .as_integer_ratio()
        )
        self.n_filt_dig = setting.n_filt_dig
        self.f_filt_dig = setting.f_filt_dig
        iir_dig_result = butter(self.n_filt_dig, 2 * self.f_filt_dig[0, :] / self.sample_rate_adc, "bandpass")
        (self.b_iir_dig, self.a_iir_dig) = iir_dig_result[0], iir_dig_result[1]
        iir_lpf_result = butter(
            self.n_filt_dig, 2 * np.array([1e-2, self.f_filt_dig[0, 0]]) / self.sample_rate_adc, "bandpass"
        )
        self.b_iir_lpf, self.a_iir_lpf = iir_lpf_result[0], iir_lpf_result[1]
        self.mem_iir_dig = np.zeros((1, len(self.b_iir_dig) - 1)) + np.power(2, self.n_bit_adc - 1)
        self.dx_sda = setting.d_xsda
        self.thr_min = setting.sda_thr_min
        if setting.d_xsda.size == 1:
            self.mode_sda = 0
        else:
            self.mode_sda = 1
        self.mem_sda = np.zeros((1, 2 * self.dx_sda[0, -1] + 1))
        self.mem_thres = np.zeros((1, 100))

        self.en_frame = 0
        self.x_delta_neg = setting.x_delta_neg
        self.x_window_length = setting.x_window_length
        self.x_delta_pos = self.x_window_length - self.x_delta_neg
        self.x_offset = setting.x_offset
        self.mem_frame = np.zeros((1, self.x_window_length))
        self.no_cluster = setting.no_cluster

    def update_iir_ana(self, fs):
        iir_ana_result = butter(self.n_filt_ana, 2 * self.f_range_ana / fs, "bandpass")
        self.b_iir_ana, self.a_iir_ana = iir_ana_result[0], iir_ana_result[1]
        self.mem_iir_ana = self.ucm + np.zeros((1, len(self.b_iir_ana) - 1))

    def pre_amp(self, uin):
        if self.realtime_mode:
            u0 = uin - self.ucm
            du0 = self.a_iir_ana @ np.array([u0, *-self.mem_iir_ana[0, :]]).T
            du1 = self.b_iir_ana @ [du0, *self.mem_iir_ana[0, :]]
            self.mem_iir_ana[0, :] = np.array(du0, self.mem_iir_ana[0])
            u_out = self.v_pri * du1 + self.ucm
            u_lfp = self.ucm
        else:
            u_out = self.ucm + self.v_pri * filtfilt(self.b_iir_ana, self.a_iir_ana, uin - self.ucm)
            u_lfp = self.ucm + self.v_pri * filtfilt(self.b_iir_lpf, self.a_iir_lpf, uin - self.ucm)
        # voltage clamping
        u_out[u_out >= self.udd] = self.udd
        u_out[u_out <= self.uss] = self.uss
        return u_out, u_lfp

    def time_delay_dig(self, uin: np.ndarray):  # review
        if uin.dtype == np.ushort:
            mat = np.ones(shape=(1, self.input_delay), dtype=np.ushort)
        else:
            mat = np.ones(shape=(1, self.input_delay))
        temp = uin[0, 0] * mat[[0], :]
        u_out = np.concatenate((temp, uin[[0], : -self.input_delay]), axis=1)
        return u_out

    def adc_nyquist(self, uin: np.ndarray, EN):
        # clamping through supply voltage
        uin_adc = uin
        uin_adc[uin >= self.u_range[1]] = self.u_range[1]
        uin_adc[uin <= self.u_range[0]] = self.u_range[0]

        uin0 = uin_adc[0, 0] + resample_poly(uin_adc[0] - uin_adc[0, 0], self.p_ratio, self.q_ratio)

        # find max index that is divisable to the sampleing rate
        max_index = int(np.floor(uin0.size / self.oversampling_ratio) * self.oversampling_ratio)
        # mean of every sub-sample window
        sub_sampled = np.mean(uin0[:max_index].reshape(-1, self.oversampling_ratio), axis=1)
        if max_index < uin0.size:
            sub_sampled = np.append(sub_sampled, np.mean(uin0[max_index:]))

        x_out, u_out = self.ad_conv_simple(sub_sampled)
        return x_out.reshape((1, -1)), u_out.reshape((1, -1))

    def dif_filt(self, xin, do_filt):
        ...

    # ---Thershold determination of neural input
    def thres(self, xin, mode) -> np.ndarray:
        if mode == 1:  # constant value
            x_out = 0 * xin + self.thr_min
        elif mode == 2:  # standard derivation of background activity
            x_out = 0 * xin + 8 * np.mean(np.abs(xin / 0.6745))
        elif mode == 3:  # Automated calculation of threshold (use by BlackRock)
            x_out = 0 * xin + 4.5 * np.sqrt(np.sum(xin**2 / len(xin)))
        elif mode == 4:  # Mean value
            x_out = 8 * self.movmean(np.array([[self.x_window_length + self.input_delay, 0]]), np.abs(xin))
        elif mode == 5:  # Lossy Peak detection
            x_out = self.hl_envelopes_idx(np.abs(xin), 21, 21)
        elif mode == 6:  # Window Mean method for Max-detection
            x_out = 0 * xin
            window_length = 20
            for i in range(1, np.floor(len(xin) / window_length)):
                x0 = np.array([[1, window_length]]) + (i - 1) * window_length
                x_out[x0[0, 0] : x0[0, 1]] = np.max(xin[x0[0, 0] : x0[0, 1]])
            x_out = 10 * self.movmean(np.array([[200, 0]]), x_out)  # Mean (Xhi,100)
        elif mode == 7:
            if xin.dtype == np.ushort:
                x0 = xin.astype(np.double)
            else:
                x0 = xin
            x_out = savgol_filter(x0, 3, 31)

        if xin.dtype == np.ushort:
            x_out = x_out.astype(np.ushort)
        else:
            x_out = x_out.astype(np.double)

        return x_out

    def spike_detection(self, xin, mode, do_sda):
        # Simple variant for direct application
        # Xsda = Xin(k + 1:end - k).^ 2 - Xin(1: end - 2 * k).*Xin(2 * k + 1: end)
        # [YPks, XPks] = findpeaks(Xsda, 'MinPeakHeight', Settings.ThresholdSDA, 'MinPeakDistance', round(500e-6 * SampleRate))
        # Methods
        # Selection is made via the vector length of dXsda
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k - NEO
        # length(x) > 1: M - TEO
        k = self.dx_sda
        if self.realtime_mode:
            # execution of realtime mode spike detection with Memory adjustment
            if do_sda:
                self.mem_sda = np.array([xin, self.mem_sda[:-1]])
                # determination of the NEO energy
                if not self.mem_sda:
                    x_sda = self.mem_sda[k + 1] ** 2 - self.mem_sda[0, 0] * self.dx_sda[-1, -1]
                else:
                    x_mteo = np.zeros(shape=k.shape)
                    for idx in range(0, k.size):
                        last_value_mem_sda = self.mem_sda[-1]
                        x_mteo[0, idx] = (
                            self.mem_sda[k[-1, -1] + 1] ** 2
                            - self.mem_sda[k[-1, -1] + 1 - k[0, idx]] * self.mem_sda[k[-1, -1] + 1 + k[0, idx]]
                        )
                    x_sda = np.max(x_mteo)
                    self.x_sda_old = x_sda
            else:
                x_sda = self.x_sda_old
        else:
            # execution of parallel spikeDetection
            if not self.mode_sda:
                # normal execution

                x_sda = xin[[0], k + 1 : -k] ** 2 - xin[[0], 0 : (-2 * k)] * xin[[0], 2 * k + 1 :]
                if xin.dtype == np.ushort:
                    mat = np.ones(shape=(1, k), dtype=np.ushort)
                else:
                    mat = np.ones(shape=(1, k))
                result_mat = mat * x_sda[0, 0]
                result_mat1 = mat * x_sda[-1, -1]
                x_sda = np.array([[*result_mat, *x_sda, *result_mat1]])
            else:
                # ---MTEO - execution
                if xin.dtype == np.ushort:
                    x_mteo = np.zeros(shape=(k.size, xin.size), dtype=np.ushort)
                else:
                    x_mteo = np.zeros(shape=(k.size, xin.size))
                for idx in range(0, k.size):
                    ksda = k[0, idx]

                    mat = np.ones(shape=(1, ksda))
                    if xin.dtype == np.ushort:
                        mat = mat.astype(np.ushort)
                    x0 = np.abs(xin[[0], ksda:-ksda] ** 2 - xin[[0], : -2 * ksda] * xin[[0], 2 * ksda :])
                    result_mat_x0 = mat * x0[0, 0]
                    result_mat1_x0 = mat * x0[-1, -1]
                    x_mteo[idx, :] = np.array([[*result_mat1_x0, *x0, *result_mat1_x0]])
                x_sda = np.max(x_mteo)
                if 0:
                    print("a plot should be drown")
                    plt.plot(x_mteo.T, label="k=1")
                    plt.plot(x_sda, label="k=3")
                    plt.legend()

            # Thershold determination
            x_thr = self.thres(x_sda, mode)

            # Trigger generation
            # xtrg = logical (x_sda >= x_thr)
            if mode == 7:
                x_thr = x_thr.astype(np.float)
                x = find_peaks(
                    x_thr,
                    "Min_Peak_Width",
                    11,
                    "Min_Peak_Distance",
                    self.x_window_length,
                    "Min_Peak_Height",
                    np.mean(x_thr + np.std(x_thr)),
                )
                x_thr = np.zeros(x_thr.shape)
                x_thr[0, x] = 1
            else:
                result = x_sda >= x_thr
                x_thr = result.astype("float32")
        return x_trg, x_sda, x_thr

    def frame_generation(self, xin: np.ndarray, x_trg):
        # check if no results are available
        if np.sum(x_trg) == 0:
            frame = np.array([[]])
            x_pos = np.array([[]])
        # Extract x- position from the trigger signal
        # x_pos = 1+ find(diff(x_trg) == 1)
        x_pos0 = find_peaks(x_trg, "Min_Peak_Distance", self.x_window_length, "Min_Peak_Height", 0.7)
        # Extract frame
        if xin.dtype == np.ushort:
            frame = np.zeros(
                shape=(len(x_pos0), self.x_window_length + self.x_offset),
                dtype=np.ushort,
            )
        else:
            frame = np.zeros(shape=(len(x_pos0), self.x_window_length + self.x_offset))
        # x_pos = np.array([[]])
        idx = 1
        for idx in range(1, len(x_pos0)):
            dx_neg = x_pos0[idx]
            dx_pos = x_pos0[idx] + self.x_window_length + self.x_offset - 1
            if dx_neg >= 1 and dx_pos <= len(x_trg):
                frame[idx, :] = xin[dx_neg:dx_pos]
                # x_pos[idx] = x_pos0[idx]

        return frame, x_pos0

    def frame_aligning(self, frame_in: np.ndarray, align_mode: int) -> np.ndarray:
        if frame_in.dtype == np.ushort:
            frame_out = np.array([]).astype(np.ushort)
        else:
            frame_out = np.array([])

        # ---Check if no results are available
        if frame_in.size == 0:
            return None

        for row in frame_in:
            frame0 = row
            frame = self.movmean(frame0, 2)

            if align_mode == 1:  # Maximum aligning
                max_pos = np.unravel_index(np.argmax(frame, axis=None), frame.shape)
            elif align_mode == 2:  # aligned to positive turning point
                max_pos = np.unravel_index(np.argmax(np.diff(frame), axis=None), np.diff(frame).shape)
            elif align_mode == 3:  # aligne to negative turning point
                max_pos = np.unravel_index(np.argmin(np.diff(frame), axis=None), np.diff(frame).shape)
                max_pos = max_pos + 1
            x = None
            x_pos0 = None
            x_pos0[x, :] = max_pos = np.array([[-self.x_delta_neg, 0, +self.x_delta_pos]])
            x_pos = x_pos0[x, :]
            if x_pos[0, 2] > len(frame0):
                state = 1
            elif x_pos[0, 0] <= 0:
                state = 2
            else:
                state = 0

            if state == 0:
                frame_out[x, :] = frame_out[x_pos[0, 0] : x_pos[0, 2] - 1]
            elif state == 1:
                if frame_in.dtype == np.ushort:
                    mat = np.ones(
                        shape=(1, np.abs(x_pos[0, 2] - len(frame0)) - 1),
                        dtype=np.ushort,
                    )
                else:
                    mat = np.ones(shape=(1, np.abs(x_pos[0, 2] - len(frame0)) - 1))
                frame_out[x, :] = np.array([[frame0[x_pos[0, 0] : -1], frame0[0, -1] * mat]])
            elif state == 2:
                if frame_in.dtype == np.ushort:
                    mat = np.ones(shape=(1, np.abs(x_pos[0, 0])), dtype=np.ushort)
                else:
                    mat = np.ones(shape=(1, np.abs(x_pos[0, 0])))

                frame_out[x, :] = np.array([[frame0[0, 0] * mat, frame0[0 : x_pos[0, 2]]]])

        return frame_out

    def fe_normal(self, frame_in):
        ...

    def fe_pca(self, frame_in):
        ...

    def clustring(
        self,
    ):
        ...

    def determine_spike_ticks(self, feat_ext, x_pos, cluster_in, time):
        ...

    def analyze_sda(self, xin, x_chk, tol):

        ...

    def ad_conv(self, uin: np.ndarray, en):

        if en:
            # ADC output + quantization
            noise_quant = (np.random.randint(1, 4, 1) - 2)[0]

            diff = uin - self.partition_adc
            negatives_idx = np.argwhere(diff <= 0)
            if negatives_idx.size > 0:
                prep = negatives_idx[0][0]
                x_out = prep - 2 ** (self.n_bit_adc - 1) - 1 + noise_quant
                u_out = self.partition_adc[prep] + (noise_quant - 1) * self.lsb / 2
                self.x_old_adc = x_out
                self.u_old_adc = u_out
        else:
            x_out = self.x_old_adc
            u_out = self.u_old_adc
        return x_out, u_out

    def ad_conv_simple(self, uin: np.ndarray):
        noise_quant = np.random.randint(1, 4, size=uin.size) - 2
        rng = self.partition_adc[-1] - self.partition_adc[0]
        x_out = np.rint((2**self.n_bit_adc - 1) * uin / rng)  # Digital value
        u_out = (x_out * rng / (2**self.n_bit_adc - 1)) + (noise_quant - 1) * self.lsb / 2
        x_out = (x_out + noise_quant).astype(np.int16)
        return x_out, u_out

    def movmean(self, arr: np.ndarray, n: int):
        result = []
        for idx, num in enumerate(arr[0]):
            sub = arr[0][max(idx - n, 0) : idx + 1]
            print(idx, sub)
            avg = np.mean(sub)
            result.append(avg)
        return np.array([result])

    def hl_envelopes_idx(self, signal: np.ndarray, dmin=1, dmax=1, split=False):
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
