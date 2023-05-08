import numpy as np
from scipy.signal import butter, filtfilt
from settings import Settings

class PreAmp:
    def __init__(self, setting: Settings, f_filt_ana):
        # --- Power supply
        self.__udd = setting.udd
        self.__uss = setting.uss
        self.__ucm = (self.__udd + self.__uss) / 2

        # --- Analogue pre-amp
        self.sample_rate_ana = setting.fs_ana
        self.__gain_ana = setting.gain_ana

        iir_spk_result = butter(setting.n_filt_ana, 2 * f_filt_ana / self.sample_rate_ana, "bandpass")
        (self.__b_iir_spk, self.__a_iir_spk) = iir_spk_result[0], iir_spk_result[1]


    # TODO: Adding noise to analogue pre-amplifier (settable)
    def pre_amp(self, uin: np.ndarray) -> np.ndarray:
        u_out = self.__ucm + self.__gain_ana * filtfilt(self.__b_iir_spk, self.__a_iir_spk, uin - self.__ucm)

        # voltage clamping
        u_out[u_out >= self.__udd] = self.__udd
        u_out[u_out <= self.__uss] = self.__uss
        return u_out

    # TODO: Implementieren (siehe MATLAB)
    def pre_amp_chopper(self, uin: np.ndarray) -> np.ndarray:
        return uin

    def time_delay(self, uin: np.ndarray, delay: float) -> np.ndarray:
        set_delay = round(delay * self.sample_rate_ana)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size-set_delay]), axis=None)
        return uout
