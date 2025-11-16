import numpy as np
from fractions import Fraction
from scipy.signal import square, resample_poly

from .adc_settings import SettingsADC
from denspp.offline.analog.common_func import CommonAnalogFunctions, CommonDigitalFunctions
from denspp.offline.analog.dev_noise import SettingsNoise, DefaultSettingsNoise, ProcessNoise


class BasicADC(CommonAnalogFunctions, CommonDigitalFunctions):
    _settings: SettingsADC
    _handler_noise: ProcessNoise

    def __init__(self, settings_dev: SettingsADC, settings_noise: SettingsNoise = DefaultSettingsNoise):
        """Basic class for applying an Analogue-Digital-Converter (ADC) on the raw data
        :param settings_dev:    Configuration class for defining properties of ADC
        :param settings_noise:  Configuration class for defining noise properties of device
        """
        super().__init__()
        self.define_voltage_range(volt_low=settings_dev.vref[1], volt_hgh=settings_dev.vref[0])
        self.define_limits(bit_signed=settings_dev.is_signed, total_bitwidth=settings_dev.Nadc, frac_bitwidth=0)
        self._handler_noise = ProcessNoise(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

        # --- Internal characteristic
        self.noise_eff_out = 0.0
        self.__dvrange = self._settings.vref[0] - self._settings.vref[1]

        # --- Resampling stuff
        (self.__p_ratio, self.__q_ratio) = (
            Fraction(self._settings.fs_adc / self._settings.fs_ana)
            .limit_denominator(100)
            .as_integer_ratio()
        )
        # --- Internal voltage values
        self.__input_snh = 0.0

    @property
    def snr_ideal(self) -> float:
        """Getting the ideal Signal-to-Noise ratio"""
        return 10 * np.log10(4) * self._settings.Nadc + 10 * np.log10(3 / 2)

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
        unoise = self._handler_noise.gen_noise_awgn_pwr(
            size = size,
            e_n=-self.snr_ideal
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
        uin_adc = self.clamp_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        uin0 += self._gen_noise(uin0.size)
        # ADC conversion
        xout = np.floor((uin0 - self._settings.vcm) / self._settings.lsb)
        xout = self.clamp_digital(xout)
        uout = self._settings.vref[1] + xout * self._settings.lsb
        # Calculating quantization error
        uerr = uin0 - uout
        return xout, uout, uerr

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
