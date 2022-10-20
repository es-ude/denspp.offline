from settings import Settings

import scipy
from scipy.signal import butter
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
        self.u_range = np.array([[0.5, 0.7]])
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
        self.b_iir_ana, self.a_iir_ana = butter(self.n_filt_ana, 2 * self.f_range_ana / fs, "bandpass")
        self.mem_iir_ana = self.ucm + np.zeros(1, len(self.b_iir_ana) - 1)
        self.input_delay = Settings.input_delay
        self.u_range = self.ucm + np.array([[-1, 1]]) * Settings.d_uref
        self.n_bit_adc = Settings.n_bit_adc
        self.lsb = np.diff(self.u_range) / 2 ^ self.n_bit_adc
        self.partition_adc = self.u_range[0, 0] + self.lsb / 2 ??????
        self.oversampling_ratio = Settings.oversampling
        self.oversampling_cnt = 1
        self.sample_rate_adc = Settings.sample_rate
        self.sample_rate_ana = fs
        self.p_ratio, self.q_ratio = sympy.nsimplify(self.sample_rate_adc* self.oversampling_ratio/ self.sample_rate_ana)
        self.n_filt_dig = Settings.n_filt_dig
        self.f_range_dig = Settings.f_filt_dig
        self.b_iir_dig , self.a_iir_dig = butter(self.n_filt_dig, 2*self.f_range_dig/self.sample_rate_adc, 'bandpass')
        self.b_iir_lpf, self.a_iir_lpf = butter(self.n_filt_dig, 2*np.array([[1e-2, self.f_range_dig[0,0]]]) /self.sample_rate_adc,'bandpass')
        self.mem_iir_dig = 2^ (self.n_bit_adc -1) + np.zeros(1, len(self.b_iir_dig) -1)
        self.dx_sda = Settings.d_xsda
        self.thr_min = Settings.sda_thr_min
        if len(Settings.d_xsda) is 1 :
            self.mode_sda = 0
        else:
            self.mode_sda = 1

        self.mem_sda = np.zeros((1, 2* self.dx_sda[]+1))???
        self.mem_thres = np.zeros((1, 100))

        self.en_frame = 0
        self.x_delta_neg = Settings.x_delta_neg
        self.x_window_length = Settings.x_window_length
        self.x_delta_pos = self.x_window_length - self.x_delta_neg
        self.x_offset = Settings.x_offset
        self.mem_frame = np.zeros((1, self.x_window_length))
        self.no_cluster = Settings.no_cluster
