import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
from settings import Settings

class ADC:
    def __init__(self, setting: Settings):
        # --- Power supply
        self.__udd = setting.udd
        self.__uss = setting.uss
        self.__ucm = (self.__udd + self.__uss) / 2

        # --- Input
        self.sample_rate_ana = setting.fs_ana

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
