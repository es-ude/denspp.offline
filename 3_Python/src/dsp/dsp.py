import numpy as np
from scipy.signal import butter, lfilter
from settings import Settings

class DSP:
    def __init__(self, setting: Settings, f_filt_dig):
        # --- Input
        self.sample_rate_adc = Settings.fs_adc
        self.oversampling = Settings.oversampling

        # --- Digital pre-processing
        iir_dig_result = butter(setting.n_filt_dig, 2 * f_filt_dig / self.sample_rate_adc, "bandpass")
        (self.__b_iir_dig, self.__a_iir_dig) = iir_dig_result[0], iir_dig_result[1]

    def time_delay(self, uin: np.ndarray, delay: float) -> np.ndarray:
        set_delay = round(delay * self.sample_rate_adc)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
        return uout

    def dig_filt_iir(self, xin: np.ndarray) -> np.ndarray:
        xout = lfilter(self.__b_iir_dig, self.__a_iir_dig, xin).astype("int16")
        return xout

    def dig_filt_fir(self, xin: np.ndarray) -> np.ndarray:
        return xin

    def dig_filt_cic(self, xin: np.ndarray) -> np.ndarray:
        return xin
