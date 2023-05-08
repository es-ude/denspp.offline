import dataclasses

import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
from settings import Settings

@dataclasses.dataclass
class SettingsADC:
    """"Individuall data class to configure the ADC"""
    def __init__(self,
        vdd:    float,
        vss:    float,
        dvref:  float,
        fs:     int,
        Nadc:   int,
        osr:    int
    ):
        self.vdd = vdd
        self.vss = vss
        self.vcm = self.__calc_vcm()
        self.vref = self.__calc_vref(dvref)
        self.sampling_rate = fs
        self.quant = Nadc
        self.osr = osr

    def __calc_vcm(self) -> float:
        return (self.vdd + self.vss)/2

    def __calc_vref(self, dvref: float) -> [float, float]:
        vrefp = self.vcm + dvref
        vrefn = self.vcm - dvref
        return [vrefp, vrefn]

class SettingsADCRecommended(SettingsADC):
    """Recommended data class to configure the ADC with standard values"""
    def __init__(self):

        super.__init__(
            vdd=0.6, vss=-0.6,
            dvref=0.2,
            fs=20000, osr=1,
            Nadc=12
        )

class ADC:
    """"Class for applying an Analogue-Digital-Converter (ADC) on the raw data for digitization"""
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

        self.offset = 0
        self.drift = 0
        self.noise = 0

    # TODO: Adding quantizazion noise (settable)
    def adc_nyquist(self, uin: np.ndarray, do_sample: bool) -> np.ndarray:
        """"Using the Nyquist Topology as an ADC"""
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
        """"Using the Successive-Approximation (SAR) Topology as an ADC"""
        raise NotImplementedError

    # TODO: Implementieren (siehe MATLAB)
    def adc_deltasigma(self, uin: np.ndarray, do_sample: bool) -> np.ndarray:
        """"Using the Delta Sigma Topology as an ADC"""
        raise NotImplementedError

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
