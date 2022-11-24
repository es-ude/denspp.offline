from fractions import Fraction
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, savgol_filter, find_peaks
from settings import Settings

class AFE:
    def __init__(self, setting: Settings):
        self.realtime_mode = setting.realtime_mode
        # --- Power supply
        self.udd = setting.udd
        self.uss = setting.uss
        self.ucm = (self.udd + self.uss) / 2

        # --- Analogue pre-amp
        self.gain_ana = setting.gain_ana
        self.sample_rate_ana = setting.desired_fs
        self.n_filt_ana = setting.n_filt_ana
        self.f_filt_ana = setting.f_filt_ana
        iir_ana_result = butter(self.n_filt_ana, 2 * self.f_filt_ana / setting.desired_fs, "bandpass")
        (self.b_iir_ana, self.a_iir_ana) = iir_ana_result[0], iir_ana_result[1]
        self.input_delay_ana = round(setting.delay_ana* self.sample_rate_ana)

        # --- ADC
        self.u_range = self.ucm + np.array([-1, 1]) * setting.d_uref
        self.n_bit_adc = setting.n_bit_adc
        self.lsb = np.diff(self.u_range) / np.power(2, self.n_bit_adc)
        self.partition_adc = self.u_range[0] + np.arange(0, np.power(2, self.n_bit_adc), 1)* self.lsb + self.lsb/2
        self.sample_rate_adc = setting.sample_rate
        self.oversampling_ratio = setting.oversampling
        (self.p_ratio, self.q_ratio) = (
            Fraction(self.sample_rate_adc * self.oversampling_ratio / self.sample_rate_ana)
            .limit_denominator(100)
            .as_integer_ratio()
        )

        # --- Digital pre-processing
        self.gain_dig = 1
        self.n_filt_dig = setting.n_filt_dig
        self.f_filt_dig = setting.f_filt_dig
        iir_dig_result = butter(self.n_filt_dig, 2 * self.f_filt_dig / self.sample_rate_adc, "bandpass")
        (self.b_iir_dig, self.a_iir_dig) = iir_dig_result[0], iir_dig_result[1]
        iir_lpf_result = butter(
            self.n_filt_dig, 2 * np.array([1e-2, self.f_filt_dig[0]]) / self.sample_rate_adc, "bandpass"
        )
        self.b_iir_lpf, self.a_iir_lpf = iir_lpf_result[0], iir_lpf_result[1]
        self.input_delay_dig = round(setting.delay_dig* self.sample_rate_adc)

        # --- Spike detection incl. thresholding and frame generation
        self.dx_sda = setting.d_xsda
        self.frame_mode = setting.mode_frame
        self.frame_length = round(setting.x_window_length* self.sample_rate_adc)
        self.frame_offset = round(setting.x_offset* self.sample_rate_adc)
        self.frame_neg = round(setting.x_window_start* self.sample_rate_adc)
        self.frame_pos = self.frame_length - self.frame_neg

        # Extension for realtime processing
        self.mem_iir_ana = self.ucm + np.zeros((1, len(self.b_iir_ana) - 1))
        self.mem_iir_dig = np.zeros((1, len(self.b_iir_dig) - 1)) + np.power(2, self.n_bit_adc - 1)
        self.mem_sda = np.zeros((1, 2* self.dx_sda[-1] + 1))
        self.mem_thres = np.zeros((1, 100))
        self.mem_frame = np.zeros((1, self.frame_length))

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
            u_out = self.gain_ana * du1 + self.ucm
            u_lfp = self.ucm
        else:
            u_out = self.ucm + self.gain_ana * filtfilt(self.b_iir_ana, self.a_iir_ana, uin - self.ucm)
            u_lfp = self.ucm + self.gain_ana * filtfilt(self.b_iir_lpf, self.a_iir_lpf, uin - self.ucm)
        # voltage clamping
        u_out[u_out >= self.udd] = self.udd
        u_out[u_out <= self.uss] = self.uss
        return u_out, u_lfp

    def time_delay_ana(self, uin: np.ndarray) -> np.ndarray:  # review
        if self.realtime_mode:
            ...
        else:
            mat = np.zeros((self.input_delay_ana,), dtype=float)
            uout = np.concatenate((mat, uin[0:uin.size-self.input_delay_ana]), axis=None)
        return uout

    def adc_nyquist(self, uin: np.ndarray, EN: bool) -> np.ndarray:
        # TODO: ADC-Funktion mit Oversampling noch einfügen
        # clamping through supply voltage
        uin_adc = uin
        uin_adc[uin >= self.u_range[1]] = self.u_range[1]
        uin_adc[uin <= self.u_range[0]] = self.u_range[0]

        if self.realtime_mode:
            x_out = self.adc_conv(uin, EN)
        else:
            # Downsampling of input
            uin0 = uin_adc[0, 0] + resample_poly(uin_adc[0] - uin_adc[0, 0], self.p_ratio, self.q_ratio)
            max_index = int(np.floor(uin0.size / self.oversampling_ratio) * self.oversampling_ratio)
            sub_sampled = np.mean(uin0[:max_index].reshape(-1, self.oversampling_ratio), axis=1)
            if max_index < uin0.size:
                sub_sampled = np.append(sub_sampled, np.mean(uin0[max_index:]))

            x_out = self.adc_conv(sub_sampled, EN)

        return x_out

    def adc_conv(self, uin: np.ndarray, en: bool):
        # en defines in realtime-mode if a conversion is running or not
        if en:
            rng = self.u_range[-1] - self.u_range[0]
            x_out = np.rint((2**self.n_bit_adc - 1) * (uin - self.ucm) / rng)  # Digital value
            noise_quant = (2*(uin - self.lsb* x_out)/self.lsb)
            x_out = (x_out + noise_quant).astype(np.int16)
            self.x_old_adc = x_out
        else:
            x_out = self.x_old_adc

        return x_out

    def time_delay_dig(self, uin: np.ndarray) -> np.ndarray:  # review
        if self.realtime_mode:
            ...
        else:
            mat = np.zeros((self.input_delay_dig,), dtype=int)
            xout = np.concatenate((mat, uin[0:uin.size-self.input_delay_dig]), axis=None)
        return xout

    def dig_filt(self, xin: np.ndarray) -> np.ndarray:
        if self.realtime_mode:
            du0 = self.a_iir_dig @ np.array([xin, *-self.mem_iir_dig[0, :]])
            du1 = self.b_iir_dig @ [du0, *self.mem_iir_dig[0, :]]
            self.mem_iir_dig[0, :] = np.array(du0, self.mem_iir_dig[0])
            xout = self.gain_dig* round(du1)
        else:
            xout = self.gain_dig* (filtfilt(self.b_iir_dig, self.a_iir_dig, xin)).astype("int16")

        return xout

    # ---Thershold determination of neural input
    # TODO: Methoden zum Thresholding kontrollieren
    def thres(self, xin: np.ndarray, mode: int) -> np.ndarray:
        if mode == 1:    # standard derivation of background activity
            x_out = 0 * xin + 8* np.mean(np.abs(xin / 0.6745))
        elif mode == 2:  # Automated calculation of threshold (use by BlackRock)
            x_out = 0 * xin + 4.5* np.sqrt(np.sum(xin**2 / len(xin)))
        elif mode == 3:  # Mean value
            x_out = 8 * self.movmean(np.array([[self.frame_length, 0]]), np.abs(xin))
        elif mode == 4:  # Lossy Peak detection
            x_out = self.hl_envelopes_idx(np.abs(xin), 21, 21)
        elif mode == 5:  # Window Mean method for Max-detection
            x_out = 0 * xin
            window_length = 20
            for i in range(1, np.floor(len(xin) / window_length)):
                x0 = np.array([[1, window_length]]) + (i - 1) * window_length
                x_out[x0[0, 0] : x0[0, 1]] = np.max(xin[x0[0, 0] : x0[0, 1]])
            x_out = 10 * self.movmean(np.array([[200, 0]]), x_out)  # Mean (Xhi,100)
        elif mode == 6: # Salvan-Goley-Fiter
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

    def spike_detection(self, xin:np.ndarray, mode: int, do_sda: bool):
        # Selection of SDA is made via the vector length of dXsda
        # length(x) == 0: applied on raw datastream
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        # length(x) > 1: M - TEO

        ksda = self.dx_sda
        if self.realtime_mode:
            # execution of realtime mode spike detection with Memory adjustment
            if do_sda:
                # TODO: SDA in Realtime noch nicht lauffähig
                self.mem_sda = np.array([xin, self.mem_sda[:-1]])
                x_sda = self.mem_sda[ksda + 1] ** 2 - self.mem_sda[0, 0] * self.dx_sda[-1, -1]
                self.x_sda_old = x_sda
            else:
                x_sda = self.x_sda_old
        else:
            if ksda.size == 0:
                x_mteo = xin
            else:
                # execution of parallel spikeDetection
                if xin.dtype == np.ushort:
                    x_mteo = np.zeros(shape=(ksda.size, xin.size), dtype=np.ushort)
                else:
                    x_mteo = np.zeros(shape=(ksda.size, xin.size))

                for idx in range(0, ksda.size):
                    ksda0 = ksda[idx]
                    x0 = np.power(xin[ksda0:-ksda0, ], 2) - xin[:-2 * ksda0, ] * xin[2 * ksda0:, ]
                    x_mteo[idx, :] = np.concatenate([x0[:ksda0, ], x0, x0[-ksda0:, ]], axis=None)

        x_sda = np.max(x_mteo, 0)
        x_thr = self.thres(x_sda, mode)

        return x_sda, x_thr

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray):
        # Trigger generation
        result = ((xsda - xthr) > 0)
        xtrg = result.astype("int")

        if self.realtime_mode:
            # TODO: Methode für Embedded einfügen und testen
            ...
        else:
            frame = np.array([[]])
            x_pos0 = np.array([[]])

            if np.sum(xtrg) == 0:
                # Abort if no results are available
                ...
            else:
                # Extract x- position from the trigger signal
                width = 5
                x_pos = np.convolve(xtrg, np.ones(width), mode="same")
                (x_pos0, _) = find_peaks(x_pos, height=1.8, distance=self.frame_length)

                frame = np.zeros(shape=(x_pos0.size, self.frame_length+self.frame_offset), dtype="int")
                # --- Generate frames
                for idx in range(0, len(x_pos0)):
                    dx_neg = x_pos0[idx]
                    dx_pos = x_pos0[idx] + self.frame_length + self.frame_offset
                    if dx_neg >= 1 and dx_pos <= len(xtrg):
                        frame0 = xraw[dx_neg:dx_pos]
                        frame[idx, :] = frame0

        return frame, x_pos0

    def frame_aligning(self, frame_in: np.ndarray, align_mode: int, en: bool) -> np.ndarray:
        # ---Check if no results are available
        if self.realtime_mode:
            # TODO: Methode für Embedded einfügen und testen
            if en:
                ...
        else:
            idx = 0
            frame_out = np.zeros(shape=(frame_in.shape[0], self.frame_length), dtype="int")
            for row in frame_in:
                frame0 = row
                frame = np.convolve(frame0, np.ones(2), mode="same")

                # --- Aligning
                if align_mode == 1:     # align to maximum
                    max_pos = np.argmax(frame, axis=None)
                elif align_mode == 2:   # align to minimum
                    max_pos = np.argmin(frame, axis=None)
                elif align_mode == 3:   # align to positive turning point
                    max_pos = np.argmax(np.diff(frame), axis=None)
                    max_pos = max_pos + 1
                elif align_mode == 4:   # align to negative turning point
                    max_pos = np.argmin(np.diff(frame), axis=None)
                    max_pos = max_pos + 1

                # --- Do the Aligning (Detection of non-full frames due to offset edges)
                dxneg = max_pos - self.frame_neg
                dxpos = max_pos + self.frame_pos

                if dxpos > len(frame0): # Missing informations at upper edge
                    mat = np.ones(shape=(1, np.abs(dxpos - len(frame0)) - 1))* frame0[-1]
                    frame1 = np.concatenate((frame0[dxneg:-1], mat), axis=None)
                elif dxneg <= 0: # Missing informations at downer edge
                    mat = np.ones(shape=(1, np.abs(dxneg)))* frame0[0]
                    frame1 = np.concatenate((mat, frame0[0:dxpos]), axis=None)
                else: # Normal state
                    frame1 = frame0[dxneg:dxpos]

                frame_out[idx,:] = frame1
                idx += 1
        return frame_out

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
