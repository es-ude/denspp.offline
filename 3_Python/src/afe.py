import numpy as np
from fractions import Fraction
from scipy.signal import butter, filtfilt, resample_poly, savgol_filter, find_peaks
from settings import Settings

# TODO: Trennung der Klassen zwischen Verarbeitung des Bitstreams i) ganzer Datensatz und ii) sample-wise
class AFE:
    def __init__(self, setting: Settings):
        # --- Power supply
        self.__udd = setting.udd
        self.__uss = setting.uss
        self.__ucm = (self.__udd + self.__uss) / 2

        # --- Analogue pre-amp
        self.__gain_ana = setting.gain_ana
        self.sample_rate_ana = setting.fs_ana
        iir_ana_result = butter(setting.n_filt_ana, 2 * setting.f_filt_ana / self.sample_rate_ana, "bandpass")
        (self.__b_iir_ana, self.__a_iir_ana) = iir_ana_result[0], iir_ana_result[1]
        self.__input_delay_ana = round(setting.delay_ana * self.sample_rate_ana)

        # --- ADC
        self.__u_range = self.__ucm + np.array([-1, 1]) * setting.d_uref
        self.__n_bit_adc = setting.n_bit_adc
        self.__lsb = np.diff(self.__u_range) / np.power(2, self.__n_bit_adc)
        self.__partition_adc = self.__u_range[0] + np.arange(0, np.power(2, self.__n_bit_adc), 1) * self.__lsb + self.__lsb / 2
        self.sample_rate_adc = setting.fs_adc
        self.__oversampling_ratio = setting.oversampling
        (self.__p_ratio, self.__q_ratio) = (
            Fraction(self.sample_rate_adc * self.__oversampling_ratio / self.sample_rate_ana)
            .limit_denominator(100)
            .as_integer_ratio()
        )

        # --- Digital pre-processing
        iir_dig_result = butter(setting.n_filt_dig, 2 * setting.f_filt_dig / self.sample_rate_adc, "bandpass")
        (self.__b_iir_dig, self.__a_iir_dig) = iir_dig_result[0], iir_dig_result[1]
        iir_lpf_result = butter(setting.n_filt_dig, 2 * np.array([1e-2, setting.f_filt_dig[0]]) / self.sample_rate_adc, "bandpass")
        self.__b_iir_lpf, self.__a_iir_lpf = iir_lpf_result[0], iir_lpf_result[1]
        self.__input_delay_dig = round(setting.delay_dig * self.sample_rate_adc)

        # --- Spike detection incl. thresholding and frame generation
        self.__dx_sda = setting.d_xsda
        self.__frame_mode = setting.mode_frame
        self.__mean_window = setting.x_window_mean
        self.frame_length = round(setting.x_window_length * self.sample_rate_adc)
        self.__frame_offset = round(setting.x_offset * self.sample_rate_adc)
        self.offset_frame = self.__frame_offset
        self.frame_neg = round(setting.x_window_start * self.sample_rate_adc)
        self.frame_pos = self.frame_length - self.frame_neg

    # TODO: Adding noise to analogue pre-amplifier (settable)
    def pre_amp(self, uin: np.ndarray) -> [np.ndarray, np.ndarray]:
        u_out = self.__ucm + self.__gain_ana * filtfilt(self.__b_iir_ana, self.__a_iir_ana, uin - self.__ucm)
        u_lfp = self.__ucm + self.__gain_ana * filtfilt(self.__b_iir_lpf, self.__a_iir_lpf, uin - self.__ucm)

        # voltage clamping
        u_out[u_out >= self.__udd] = self.__udd
        u_out[u_out <= self.__uss] = self.__uss
        return u_out, u_lfp

    # TODO: Implementieren (siehe MATLAB)
    def pre_amp_chopper(self, uin: np.ndarray) -> [np.ndarray, np.ndarray]:
        return uin, uin

    def time_delay_ana(self, uin: np.ndarray) -> np.ndarray:  # review
        mat = np.zeros(shape=(self.__input_delay_ana,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size-self.__input_delay_ana]), axis=None)
        return uout

    # TODO: Adding quantizazion noise (settable)
    def adc_nyquist(self, uin: np.ndarray, do_sample: bool) -> np.ndarray:
        # TODO: ADC-Funktion mit Oversampling noch einfügen
        # clamping through supply voltage
        uin_adc = uin
        uin_adc[uin > self.__u_range[1]] = self.__u_range[1]
        uin_adc[uin < self.__u_range[0]] = self.__u_range[0]

        uin0 = uin_adc[0] + resample_poly(uin_adc - uin_adc[0], self.__p_ratio, self.__q_ratio)
        max_index = int(np.floor(uin0.size / self.__oversampling_ratio) * self.__oversampling_ratio)
        sub_sampled = np.mean(uin0[:max_index].reshape(-1, self.__oversampling_ratio), axis=1)
        if max_index < uin0.size:
            sub_sampled = np.append(sub_sampled, np.mean(uin0[max_index:]))

        x_out = self.__adc_conv(sub_sampled, do_sample)

        return x_out

    # TODO: Implementieren (siehe MATLAB)
    def adc_sar(self, uin: np.ndarray, do_sample: bool) -> np.ndarray:
        return uin

    # TODO: Implementieren (siehe MATLAB)
    def adc_deltasigma(self, uin: np.ndarray, do_sample: bool) -> np.ndarray:
        return uin

    def __adc_conv(self, uin: np.ndarray, do_sample: bool):
        if do_sample:
            rng = self.__u_range[-1] - self.__u_range[0]
            x_out = np.rint((2**self.__n_bit_adc - 1) * (uin - self.__ucm) / rng)  # Digital value
            noise_quant = (2*(uin - self.__lsb * x_out)/self.__lsb)
            x_out = (x_out + noise_quant).astype(np.int16)
            self.__x_old_adc = x_out
        else:
            x_out = self.__x_old_adc

        return x_out

    def time_delay_dig(self, uin: np.ndarray) -> np.ndarray:  # review
        mat = np.zeros(shape=(self.__input_delay_dig,), dtype=int)
        xout = np.concatenate((mat, uin[0:uin.size-self.__input_delay_dig]), axis=None)
        return xout

    def dig_filt_iir(self, xin: np.ndarray) -> np.ndarray:
        xout = filtfilt(self.__b_iir_dig, self.__a_iir_dig, xin).astype("int16")
        return xout

    # TODO: Implementieren (siehe MATLAB)
    def dig_filt_fir(self, xin: np.ndarray) -> np.ndarray:
        return xin

    def dig_filt_cic(self, xin: np.ndarray) -> np.ndarray:
        return xin

    # ---Thershold determination of neural input
    def thres(self, xin: np.ndarray, mode: int) -> np.ndarray:
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

    def spike_detection(self, xin: np.ndarray, mode: int, do_sda: bool) -> [np.ndarray, np.ndarray]:
        # Selection of SDA is made via the vector length of dXsda
        # length(x) == 0: applied on raw datastream
        # length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        # length(x) > 1: M - TEO

        ksda = self.__dx_sda
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

    def frame_generation(self, xraw: np.ndarray, xsda: np.ndarray, xthr: np.ndarray) -> [np.ndarray, np.ndarray]:
        # Trigger generation
        result = ((xsda - xthr) > 0)
        xtrg = result.astype("int")

        frame = np.array([[]])
        x_pos0 = np.array([[]])

        if np.sum(xtrg) == 0:
            # Abort if no results are available
            ...
        else:
            # Extract x- position from the trigger signal
            width = 5
            x_pos = np.convolve(xtrg, np.ones(width), mode="same")
            (x_pos0, _) = find_peaks(x_pos, distance=self.frame_length)

            lgth_frame = self.frame_length + 2 * self.__frame_offset
            frame = np.zeros(shape=(x_pos0.size, lgth_frame), dtype="int")
            # --- Generate frames
            idx = 0
            for pos_frame in x_pos0:
                dx_neg = pos_frame
                dx_pos = pos_frame + lgth_frame
                frame[idx, :] = xraw[dx_neg:dx_pos]
                idx += 1

        return frame, x_pos0

    def frame_aligning(self, frame_in: np.ndarray, align_mode: int) -> np.ndarray:
        # ---Check if no results are available
        idx = 0
        frame_out = np.zeros(shape=(frame_in.shape[0], self.frame_length), dtype="int")
        for row in frame_in:
            frame0 = row
            frame = np.convolve(frame0, np.ones(2), mode="same")
            # --- Window of finding feature
            idx0 = int(self.__frame_offset)
            idx1 = row.size - idx0

            # --- Aligning
            # TODO: Fenster-Methode einfügen
            search_frame = frame #[idx0:idx1]
            if align_mode == 1:     # align to maximum
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

    def calculate_snr(self, yin: np.ndarray, ymean: np.ndarray):
        A = np.sum(np.square(yin))
        B = np.sum(np.square(ymean - yin))
        outdB = 10 * np.log10(A/B)
        return outdB

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
