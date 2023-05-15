import numpy as np
from src.adc.adc_basic import ADC_Basic, SettingsADC, SettingsNon, RecommendedSettingsNon

class ADC_Nyquist(ADC_Basic):
    """Class for applying a Nyquist Analogue-Digital-Converter (ADC) on the raw data"""
    def __init__(self, setting: SettingsADC):
        super().__init__(setting, RecommendedSettingsNon())
        # --- Transfer function
        self.__lsb = self.settings.lsb
        self.__dvrange = 2 * self.settings.dvref
        self.__partition_digital = np.arange(0, 2 ** self.settings.Nadc, 1)
        self.__partition_digital -= 2 ** (self.settings.Nadc - 1) if self.settings.type_out == "signed" else 0
        self.__partition_voltage = np.arange(0, 2 ** self.settings.Nadc, 1) * self.__lsb
        self.__partition_voltage += self.settings.vref[1] + self.settings_non.offset + self.__lsb / 2

    def __adc_conv_sample(self, uin: float) -> np.ndarray:
        """Converting the value (nyquist ideal, sample converting)"""
        x0 = np.where(uin <= self.__partition_voltage)
        xout = self.__partition_digital[x0[0][0]]
        return xout

    def __adc_conv_stream(self, uin: np.ndarray) -> np.ndarray:
        """Converting the value (nyquist ideal, stream converting)"""
        x_out = np.zeros(shape=uin.shape)
        for idx, vol in enumerate(uin):
            x_out[idx] = self.__adc_conv_sample(vol)
        return self.clipping_digital(x_out).astype(np.int)

    def adc_nyquist(self, uin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Using the Nyquist Topology as an ADC
        input:
        uin     - Input voltage
        output:
        x_out   - Output digital value
        quant_er - Quantization error
        """
        # Do resampling and conversion
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        x_out = self.__adc_conv_stream(uin0)
        # Add noise and calc quantization error
        quant_err = uin0 - x_out * self.__lsb
        x_out += self.gen_noise(uin0.size).astype(np.int)
        return x_out, quant_err
