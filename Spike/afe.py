from settings import Settings

import scipy
from scipy.signal import butter, filtfilt, resample

import os
import numpy as np
import sympy


class AFE:
    def __init__(self, setting: Settings, fs, realtime_mode):

        self.realtime_mode = 0
        #  --- Power Supply
        # if settings.Udd != None:
        #     self.UDD = settings.Udd
        # else:
        self.udd = 1.2
        self.ucm = 0.6
        self.uss = 0
        #  --- Variables for pre-amplification
        self.v_pri = 40
        self.n_filt_ana = 2
        self.f_range_ana = np.array([[1, 5e3]])
        # self.a_iir_ana = None
        # self.b_iir_ana = None
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

        self.ucm = (self.udd + self.uss) / 2
        self.v_pri = Settings.gain_pre
        self.n_filt_ana = Settings.n_filt_ana
        self.f_filt_ana = Settings.f_filt_ana
        iir_ana_result = butter(self.n_filt_ana, 2 * self.f_range_ana / fs, "bandpass")
        self.b_iir_ana, self.a_iir_ana = iir_ana_result[0], iir_ana_result[1]
        self.mem_iir_ana = self.ucm + np.zeros((1, len(self.b_iir_ana) - 1))
        self.input_delay = Settings.input_delay

        self.u_range = self.ucm + np.array([-1, 1]) * Settings.d_uref
        self.n_bit_adc = Settings.n_bit_adc
        self.lsb = np.diff(self.u_range) / 2 ^ self.n_bit_adc
        self.partition_adc = np.arange(self.u_range[0] + self.lsb / 2, self.u_range[1] + self.lsb / 2, self.lsb)
        self.oversampling_ratio = Settings.oversampling
        self.oversampling_cnt = 1
        self.sample_rate_adc = Settings.sample_rate
        self.sample_rate_ana = fs
        self.p_ratio, self.q_ratio = sympy.nsimplify(
            self.sample_rate_adc * self.oversampling_ratio / self.sample_rate_ana
        )
        self.n_filt_dig = Settings.n_filt_dig
        self.f_range_dig = Settings.f_filt_dig
        iir_dig_result = butter(self.n_filt_dig, 2 * self.f_range_dig / self.sample_rate_adc, "bandpass")
        self.b_iir_dig, self.a_iir_dig = iir_dig_result[0], iir_dig_result[1]
        iir_lpf_result = butter(
            self.n_filt_dig, 2 * np.array([[1e-2, self.f_range_dig[0, 0]]]) / self.sample_rate_adc, "bandpass"
        )
        self.b_iir_lpf, self.a_iir_lpf = iir_lpf_result[0], iir_lpf_result[1]
        self.mem_iir_dig = 2 ^ (self.n_bit_adc - 1) + np.zeros((1, len(self.b_iir_dig) - 1))
        self.dx_sda = Settings.d_xsda
        self.thr_min = Settings.sda_thr_min
        if len(Settings.d_xsda) is 1:
            self.mode_sda = 0
        else:
            self.mode_sda = 1

        self.mem_sda = np.zeros((1, 2 * self.dx_sda[-1] + 1))
        self.mem_thres = np.zeros((1, 100))

        self.en_frame = 0
        self.x_delta_neg = Settings.x_delta_neg
        self.x_window_length = Settings.x_window_length
        self.x_delta_pos = self.x_window_length - self.x_delta_neg
        self.x_offset = Settings.x_offset
        self.mem_frame = np.zeros((1, self.x_window_length))
        self.no_cluster = Settings.no_cluster

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
            u_out = self.ucm + self.v_pri @ filtfilt(self.b_iir_ana, self.a_iir_ana, uin - self.ucm)
            u_lfp = self.ucm + self.v_pri @ filtfilt(self.b_iir_lpf, self.a_iir_lpf, uin - self.ucm)
        # voltage clamping
        u_out[u_out >= self.udd] = self.udd
        u_out[u_out <= self.uss] = self.uss
        return u_out, u_lfp

    def time_delay_dig(self, uin: np.ndarray):  # review
        if uin.dtype == np.ushort:
            mat = np.ones(shape=(1, self.input_delay), dtype=np.ushort)
        else:
            mat = np.ones(shape=(1, self.input_delay))
        temp = uin[0] @ mat
        u_out = np.array([*temp, *uin[: -self.input_delay]])
        return u_out

    def adc_nyquist(self, uin: np.ndarray, EN):
        # clamping through supply voltage

        uin_adc = uin
        uin_adc[uin >= self.u_range[1]] = self.u_range[1]
        uin_adc[uin <= self.u_range[0]] = self.u_range[0]

        uin0 = uin_adc[0] + resample(uin_adc - uin_adc[0], self.p_ratio, self.q_ratio)
        idx = 1
        x0 = 1
        x1 = self.oversampling_ratio
        x_out = np.zeros(shape=uin0.size / self.oversampling_ratio)
        u_out = None
        while x0 <= len(uin0):
            if self.oversampling_ratio == 1:
                x_out[idx], u_out[idx] = self.ad_conv(uin0[idx], 1)
            else:
                if x1 > len(uin0):
                    x1 = len(uin0)
                x_out[idx], u_out[idx] = self.ad_conv(np.mean(uin0[x0:x1]), 1)
            x0 = x0 + self.oversampling_ratio
            x1 = x1 + self.oversampling_ratio
            idx = idx + 1

        return x_out, u_out

    def dif_filt(self, xin, do_filt):
        ...

    def Thres(self, xin, mode):
        ...

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

                x_sda = xin[k + 1 : -k] ** 2 - xin[1 : (-2 * k)] * xin[2 * k + 1 :]
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
                    x0 = np.abs(xin[ksda + 1 : -ksda] ** 2 - xin[: -2 * ksda] * xin[2 * ksda + 1 :])
                    result_mat_x0 = mat * x0[0, 0]
                    result_mat1_x0 = mat * x0[-1, -1]
                    x_mteo[idx, :] = np.array([[*result_mat1_x0, *x0, *result_mat1_x0]])
                x_sda = np.max(x_mteo)
                if 0:
                    print("a plot should be drown")

        return x_trg, x_sda, x_thr

    def frame_generation(self, xin, x_trg):
        ...

    def frame_aligning(self, frame_in, align_mode):
        ...

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
            noise_quant = np.random.uniform(1, 3, 1) - 2
            diff = uin - self.partition_adc
            negatives_idx = np.argwhere(diff <= 0)
            if negatives_idx.size > 0:
                prep = negatives_idx[0]
                x_out = prep - 2 ** (self.n_bit_adc - 1) - 1 + noise_quant
                u_out = self.partition_adc[prep] + (noise_quant - 1) * self.lsb / 2
                self.x_old_adc = x_out
                self.u_old_adc = u_out
        else:
            x_out = self.x_old_adc
            u_out = self.u_old_adc
        return x_out, u_out
