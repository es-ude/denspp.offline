import numpy as np
from fractions import Fraction
from scipy.signal import square, resample_poly

from .adc_settings import SettingsADC
from denspp.offline.analog.common_func import CommonAnalogFunctions, CommonDigitalFunctions
from denspp.offline.analog.dev_noise import SettingsNoise, RecommendedSettingsNoise, ProcessNoise


class BasicADC(CommonAnalogFunctions, CommonDigitalFunctions):
    handler_noise: ProcessNoise

    def __init__(self, settings_adc: SettingsADC, settings_noise: SettingsNoise = RecommendedSettingsNoise):
        """Basic class for applying an Analogue-Digital-Converter (ADC) on the raw data
        :param settings_adc: SettingsADC
        :param settings_noise: SettingsNoise
        """
        CommonAnalogFunctions(settings_adc)
        CommonDigitalFunctions(settings_adc)

        self.handler_noise = ProcessNoise(settings_noise, settings_adc.fs_ana)
        self._settings = settings_adc

        # --- Internal characteristic
        self.noise_eff_out = 0.0
        self.__dvrange = self._settings.vref[0] - self._settings.vref[1]
        self.__lsb = self._settings.lsb
        self.__oversampling_ratio = self._settings.osr
        self.__snr_ideal = 10 * np.log10(4) * self._settings.Nadc + 10 * np.log10(3 / 2)
        self._digital_border = np.array([0, 2 ** self._settings.Nadc - 1])
        self._digital_border -= 2 ** (self._settings.Nadc - 1) if self._settings.type_out == "signed" else 0
        # --- Resampling stuff
        (self.__p_ratio, self.__q_ratio) = (
            Fraction(self._settings.fs_adc / self._settings.fs_ana)
            .limit_denominator(100)
            .as_integer_ratio()
        )
        # --- Internal voltage values
        self.__input_snh = 0.0

    def __do_snh_sample(self, uin: np.ndarray, do: bool | np.ndarray) -> np.ndarray:
        """Performing sample-and-hold (S&H) stage for buffering input value"""
        uout = uin
        if do:
            uout = self.__input_snh
            self.__input_snh = uin
        return uout

    def do_snh_stream(self, uin: np.ndarray, f_snh: float) -> np.ndarray:
        """Performing sample-and-hold (S&H) stage for buffering input value"""
        t = np.arange(0, uin.size, 1) / self._settings.fs_ana
        clk_fsh = square(2 * np.pi * t * f_snh, duty=0.5)
        do_snh = np.where(np.diff(clk_fsh) >= 0.5)
        do_snh += 1

        uout = np.zeros(shape=uin.shape)
        for idx, do_snh in enumerate(do_snh):
            uout[idx] = self.__do_snh_sample(uin[idx], do_snh)
        return uout

    def _do_resample(self, uin: np.ndarray) -> np.ndarray:
        """Do resampling of input values"""
        if uin.size == 1:
            uout = uin
        else:
            uout = uin[0] + resample_poly(uin - uin[0], self.__p_ratio, self.__q_ratio)
        return uout

    def _gen_noise(self, size: int) -> np.ndarray:
        """Generate the transient input noise of the amplifier"""
        unoise = self.handler_noise.gen_noise_awgn_pwr(
            size = size,
            e_n=-self.__snr_ideal
        )
        return unoise

    def adc_ideal(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Using the ideal ADC
        Args:
            uin:    Input voltage
        Returns:
            Tuple with three numpy arrays [x_out = Output digital value, u_out = Output digitized voltage, uerr = Quantization error]
        """
        # Pre-Processing
        uin_adc = self.voltage_clipping(uin)
        uin0 = self._do_resample(uin_adc)
        uin0 += self._gen_noise(uin0.size)
        # ADC conversion
        xout = np.floor((uin0 - self._settings.vcm) / self.__lsb)
        xout = self.digital_clipping(xout)
        uout = self._settings.vref[1] + xout * self.__lsb
        # Calculating quantization error
        uerr = uin0 - uout
        return xout, uout, uerr

    def _do_downsample(self, uin: np.ndarray) -> np.ndarray:
        """Performing a simple downsampling of the adc data stream"""
        (p_ratio, q_ratio) = (
            Fraction(self._settings.fs_dig / self._settings.fs_adc)
            .limit_denominator(100)
            .as_integer_ratio()
        )
        uout = uin[0] + resample_poly(uin - uin[0], p_ratio, q_ratio)
        return uout

    def do_cic(self, uin: np.ndarray, num_stages: int=5) -> np.ndarray:
        """Performing the CIC filter at the output of oversampled ADC"""
        output_transient = list()
        gain = (self._settings.osr * 1) ** num_stages

        class integrator:
            def __init__(self):
                self.yn = 0
                self.ynm = 0

            def update(self, inp):
                self.ynm = self.yn
                self.yn = (self.ynm + inp)
                return (self.yn)

        class comb:
            def __init__(self):
                self.xn = 0
                self.xnm = 0

            def update(self, inp):
                self.xnm = self.xn
                self.xn = inp
                return (self.xn - self.xnm)

        ## Generate Integrator and Comb lists (Python list of objects)
        intes = [integrator() for a in range(num_stages)]
        combs = [comb() for a in range(num_stages)]

        ## Performing Decimation CIC Filter
        for (s, v) in enumerate(uin):
            z = v
            for i in range(num_stages):
                z = intes[i].update(z)

            if (s % self._settings.osr) == 0:  # decimate is done here
                for c in range(num_stages):
                    z = combs[c].update(z)
                    j = z
                output_transient.append(j / gain)  # normalise the gain
        return np.array(output_transient)

    @staticmethod
    def _generate_sar_empty_data(shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uout = np.zeros(shape=shape, dtype=np.float32)
        xout = np.zeros(shape=shape, dtype=np.int16)
        uerr = np.zeros(shape=shape, dtype=np.float32)
        return xout, uout, uerr

    @staticmethod
    def _generate_dsigma_empty_data(shape) -> tuple[np.ndarray, np.ndarray]:
        xout_hs = np.zeros(shape=shape, dtype=np.int32)
        xbit = np.zeros(shape=shape, dtype=np.int32)
        return xout_hs, xbit

    @staticmethod
    def do_decimation_polyphase_order_one(uin: np.ndarray) -> np.ndarray:
        """Performing first order Non-Recursive Polyphase Decimation on input"""
        last_sample_hs = 0
        uout = []
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_hs)
            last_sample_hs = val

        uout = np.array(uout)
        return uout

    @staticmethod
    def do_decimation_polyphase_order_two(uin: np.ndarray) -> np.ndarray:
        """Performing second order Non-Recursive Polyphase Decimation on input"""
        last_sample_hs = 0
        last_sample_ls = 0
        uout = []
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_ls + 2 * last_sample_hs)
                last_sample_ls = val
            last_sample_hs = val

        uout = np.array(uout)
        return uout
