import numpy as np
from package.metric import calculate_snr
from package.data_process.process_noise import noise_awgn


class EasyNeuralData_Generator:
    def __init__(self, no_spk: float, fs: float):
        self.__fs = fs
        self.__no_spk = no_spk
        self.__t_blend = None
        self.__t_active = None
        self.__t_sim = None
        self.__t_size = None
        self.time = None
        self.active_pos = None
        self.__num_rpt_spk = 200

    def gen_spike_activity(self, amp: float, period: float, firing_rate: float, snrdB: float, num_spk_blend=2) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Generate an artificial bitstream of spike activity"""
        self.__t_blend = float(num_spk_blend / firing_rate)
        self.__t_active = float(self.__no_spk / firing_rate)
        self.__t_sim = 2 * self.__t_blend + self.__t_active
        self.__t_size = int(self.__t_sim * self.__fs)
        self.time = np.linspace(0, self.__t_sim, num=self.__t_size, endpoint=True, dtype=float)

        spk_wave, noise_pwr = self.__gen_spike_shape_amp(amp, period, snrdB)
        spk_size = spk_wave.size

        # --- Generate spike train
        spk_u = 0 * self.time
        spk_off = int(self.__fs * self.__t_blend)
        spk_no = int(self.__t_active * firing_rate)
        spk_dx = int(self.__fs / firing_rate)
        spk_pos = spk_off + spk_size + spk_dx * np.arange(0, spk_no)
        spk_max_num = 2 + int(np.floor(self.__t_sim / period))

        for idx in spk_pos:
            spk_u[idx : idx + spk_size] = spk_wave

        spk_pos_avai = spk_pos + int(np.round(spk_size/2))
        spk_pos_may = spk_size * np.arange(1, spk_max_num)
        spk_u += noise_awgn(self.__t_size, self.__fs, noise_pwr)[0]
        self.active_pos = [0+spk_off, self.time.size-spk_off]

        return spk_u, spk_pos_avai, spk_pos_may

    def gen_lfp(self, amp: float, freq: float, no_harm: int) -> np.ndarray:
        """Generate an artificial bitstream of the local field potential"""
        lfp = 0 * self.time
        # --- Signal generation
        for n in np.arange(0, no_harm):
            lfp += np.sin(2 * np.pi * freq * (n+1) * self.time) * np.exp(-0.6 * (n ** 2))
        # --- Amplitude generation
        lfp = amp * lfp / np.max(lfp)
        return lfp

    def cut_frames(self, signal: np.ndarray, xpos: np.ndarray, window: float) -> np.ndarray:
        """Cutting the spike frames out of the transient signal"""
        window_size = int(window * self.__fs / 2)
        spk_frames = np.zeros(shape=(xpos.size, 2*window_size))
        for idx, pos in enumerate(xpos):
            spk_frames[idx, :] = signal[pos-window_size:pos+window_size]

        return spk_frames

    # --- Functions for generating noise
    # --- Functions for generating spike waveforms
    def __gen_spike_shape_snr(self, amp: float, period: float, snrdB: float) -> [np.ndarray, np.ndarray]:
        """Generate an artificial shape waveform without noise"""
        ratio_exp_sine = 0.66
        Nspk = int(period * self.__fs)

        spk = -ratio_exp_sine * self.__signal_gaussian(Nspk, 0.23, 0.06)
        spk += (1 - ratio_exp_sine) * self.__signal_sine(Nspk, 0.35, 1)
        spk = amp * spk / ratio_exp_sine

        # --- Adding noise
        SNR_soll = snrdB
        SNR_diff = 100
        noise_pwr = -80
        while (np.abs(SNR_diff) > 0.05):
            snr_calc = np.zeros(shape=(self.__num_rpt_spk,))
            for idx in snr_calc:
                noise = noise_awgn(Nspk, self.__fs, noise_pwr)[0]
                snr_calc[idx] = calculate_snr(spk + noise, spk)
            SNR_ist = np.mean(snr_calc)
            SNR_diff = SNR_ist - SNR_soll
            noise_pwr += 0.25 * SNR_diff / 10

        return spk, noise_pwr

    def __gen_spike_shape_amp(self, noise_pwr: float, period: float, snrdB: float) -> [np.ndarray, np.ndarray]:
        """Generate an artificial shape waveform without noise"""
        ratio_exp_sine = 0.66
        Nspk = int(period * self.__fs)

        spk0 = -ratio_exp_sine * self.__signal_gaussian(Nspk, 0.23, 0.06)
        spk0 += (1 - ratio_exp_sine) * self.__signal_sine(Nspk, 0.35, 1)
        spk0 = spk0 / ratio_exp_sine

        # --- Adding noise
        SNR_soll = snrdB
        SNR_diff = 100
        amp0 = 1e-9
        while(np.abs(SNR_diff) > 0.01):
            spk = amp0 * spk0
            snr_calc = np.zeros(shape=(self.__num_rpt_spk,))
            for idx, snr in enumerate(snr_calc):
                noise = noise_awgn(Nspk, self.__fs, noise_pwr)[0]
                snr_calc[idx] = calculate_snr(spk + noise, spk)
            SNR_ist = np.mean(snr_calc)
            SNR_diff = SNR_ist - SNR_soll
            amp0 = amp0 / np.sqrt(np.power(10, SNR_diff/10))

        return spk, noise_pwr

    def __signal_gaussian(self, size: int, mu: float, sigma: float) -> np.ndarray:
        """Signal waveform: Gaussian"""
        x = np.linspace(0, 1, size)
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def __signal_sine(self, size: int, mu: float, sigma: float) -> np.ndarray:
        """Signal waveform: Positive sine waveform """
        x = np.linspace(0, 1, size)
        y = np.sin(2 * np.pi * (x - mu) / sigma)

        # --- Clipping
        dx = int(size * mu) + np.array([0, 1 + size/2 * sigma], dtype=int)
        if dx[1] > size:
            dx[1] = size-1
        y[y < 0] = 0
        y[0:dx[0]] = 0
        y[dx[1]:] = 0
        return y
