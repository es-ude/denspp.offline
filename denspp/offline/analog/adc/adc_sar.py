import numpy as np
from .adc_basic import BasicADC
from .adc_settings import SettingsADC, SettingsNon, RecommendedSettingsNon
from denspp.offline.analog.dev_noise import ProcessNoise


class SuccessiveApproximation(BasicADC):
    _settings: SettingsADC
    _handler_noise: ProcessNoise

    def __init__(self, settings_dev: SettingsADC, settings_non: SettingsNon = RecommendedSettingsNon) -> None:
        """"Class for applying a Successive Approximation (SAR) Analogue-Digital-Converter (ADC) on the raw data
            :param settings_dev:    Configuration class for defining properties of ADC
            :param settings_non:    Configuration class for non-idealities / parasitics of ADC (next feature)
        """
        super().__init__(settings_dev)
        self.__use_noise = settings_non.use_noise
        # --- Transfer function
        self.__partition_digital = 2 ** np.arange(0, self._settings.Nadc)
        self.__partition_voltage = (self.__partition_digital / 2 ** self._settings.Nadc) * self._settings.vref_range
        self.__type_offset = [2 ** (self._settings.Nadc - 1) if self._settings.is_signed else 0]
        # --- Internal signals for noise shaping
        self.alpha_int = [1.0, 0.5]
        self.__stage_one_dly = self._settings.vcm
        self.__stage_two_dly = self._settings.vcm

    def __adc_sar_sample(self, uin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Running the SAR on input data"""
        # --- Bitmask generation
        BitMask = np.zeros(shape=(self._settings.Nadc,), dtype=int)
        BitMask[-1] = 1
        # --- Run SAR code
        for idx in range(0, self._settings.Nadc):
            uref = self._settings.vref[1] + np.sum(BitMask * self.__partition_voltage)
            BitMask[self._settings.Nadc - 1 - idx] = np.heaviside(uin - uref, 1)
            if not idx == self._settings.Nadc - 1:
                BitMask[self._settings.Nadc - 2 - idx] = 1

        uout = self._settings.vref[1] + np.sum(BitMask * self.__partition_voltage)
        xout = (np.sum(BitMask * self.__partition_digital) - self.__type_offset).astype(int)
        return uout, xout

    def adc_sar(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the SAR Topology as an ADC
        Args:
            uin:    Input voltage
        Returns:
            Tuple with three numpy arrays [x_out = Output digital value, u_out = Output digitized voltage, quant_er = Quantization error]
        """
        # Resampling of input
        uin_adc = self.clamp_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        unoise = self._gen_noise(uin0.size) if self.__use_noise else np.zeros(shape=uin0.shape)

        # Running SAR code
        xout, uout, uerr = self._generate_sar_empty_data(uin0.shape)
        for idx, umod in enumerate(uin0):
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self._settings.lsb)
            uerr[idx] = umod - uout[idx]
        return xout, uout, uerr

    def adc_sar_ns_delay(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (Delay of last sample)
        Args:
            uin:    Input voltage
        Returns:
            Tuple with three numpy arrays [x_out = Output digital value, u_out = Output digitized voltage, quant_er = Quantization error]
        """
        # Resampling of input
        uin_adc = self.clamp_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        unoise = self._gen_noise(uin0.size) if self.__use_noise else np.zeros(shape=uin0.shape)
        # Running SAR code
        xout, uout, uerr = self._generate_sar_empty_data(uin0.shape)
        for idx, din in enumerate(uin0):
            umod = din + self.__stage_one_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self._settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Noise shaping post-processing
            self.__stage_one_dly = uerr[idx]
        return xout, uout, uerr

    def adc_sar_ns_order_one(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (First order with integration)
        Args:
            uin:    Input voltage
        Returns:
            Tuple with three numpy arrays [x_out = Output digital value, u_out = Output digitized voltage, quant_er = Quantization error]
        """
        # Resampling of input
        uin_adc = self.clamp_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        unoise = self._gen_noise(uin0.size) if self.__use_noise else np.zeros(shape=uin0.shape)
        # Running SAR code
        xout, uout, uerr = self._generate_sar_empty_data(uin0.shape)
        for idx, din in enumerate(uin0):
            # --- SAR processing
            umod = din + self.__stage_one_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self._settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Post-processing: Noise shaping
            self.__stage_one_dly += self.alpha_int[0] * uerr[idx]
        return xout, uout, uerr

    def adc_sar_ns_order_two(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (Second order with integration)
        Args:
            uin:    Input voltage
        Returns:
            Tuple with three numpy arrays [x_out = Output digital value, u_out = Output digitized voltage, quant_er = Quantization error]
        """
        # Resampling of input
        uin_adc = self.clamp_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        unoise = self._gen_noise(uin0.size) if self.__use_noise else np.zeros(
            shape=uin0.shape)
        # Running SAR code
        xout, uout, uerr = self._generate_sar_empty_data(uin0.shape)
        for idx, din in enumerate(uin0):
            umod = din + self.__stage_two_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self._settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Noise shaping post-processing
            self.__stage_one_dly += self.alpha_int[0] * uerr[idx]
            self.__stage_two_dly += self.alpha_int[1] * self.__stage_one_dly
        return xout, uout, uerr
