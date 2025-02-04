import numpy as np

from .adc_basic import BasicADC
from denspp.offline.analog.adc import SettingsADC, SettingsNon, RecommendedSettingsNon


class DeltaSigmaADC(BasicADC):
    """Class for using Continuous Time Delta Sigma ADC"""
    def __init__(self, settings_adc: SettingsADC, settings_non=RecommendedSettingsNon):
        super().__init__(settings_adc)
        # --- Internal variables
        self.use_noise = False
        self.__dac_order = 2
        self.__dac_dvrange = self._settings.vref[0] - self._settings.vref[1]
        self.__dac_lsb = self.__dac_dvrange / 2 ** self.__dac_order
        self.__partition_digital = np.arange(0, 2 ** self.__dac_order, 1) / 2 ** self.__dac_order
        self.__partition_voltage = self._settings.vref[1] + self.__partition_digital * self.__dac_dvrange + self.__dac_lsb / 2

        # --- Variables for post-processing (noise-shaping)
        self.__stage_one_dly = self._settings.vcm
        self.__stage_two_dly = self._settings.vcm

    def __ds_modulator(self, uin: np.ndarray, ufb: np.ndarray) -> np.ndarray:
        """Performing first order delta sigma modulator
        inputs:
        uin     - input voltage
        ufb     - feedback voltage
        output:
        du      - difference voltage
        """
        du = uin - ufb
        # Voltage clipping
        du = du if not du > self._settings.vdd else self._settings.vdd
        du = du if not du < self._settings.vss else self._settings.vss
        # Output
        return du

    @staticmethod
    def __stream_converter(xin: int) -> np.ndarray:
        """Performing the stream conversion"""
        xout = (1 + np.sum((-1) ** (1 - xin))) / 2
        return xout

    def __comp_1bit(self, uin: float) -> [np.ndarray, np.ndarray]:
        """1-bit DAC for DS modulation"""
        xout = np.heaviside(uin - self._settings.vcm, 1)
        ufb = self._settings.vref[0] if xout == 1 else self._settings.vref[1]
        return xout, ufb

    def __comp_Nbit(self, uin: float) -> [np.ndarray, np.ndarray]:
        """N-bit DAC for DS modulation"""
        input = uin * np.ones(shape=self.__partition_voltage.shape)
        result = np.heaviside(input - self.__partition_voltage, 1)
        xout = np.sum(result)
        ufb = self._settings.vref[1] + xout * self.__dac_lsb

        return xout, ufb

    def adc_deltasigma_order_one(self, uin: np.ndarray) -> np.ndarray:
        """"Using the Delta Sigma Topology as an ADC (1-bit, first order)
        Args:
            uin: Input voltage [V]
        Returns:
            Corresponding digitized data value from Delta-Sigma ADC
        """
        # Resampling the input to sampling frequency of ADC with oversampling
        uin_adc = self.voltage_clipping(uin)
        uin0 = self._do_resample(uin_adc)
        uin0 += self._gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)

        # Running the delta sigma modulator
        xout_hs, xbit = self._generate_dsigma_empty_data(uin0.shape)
        umod_one = self._settings.vcm
        ufb0 = self._settings.vref[1]
        # --- DS Modulator (at high frequency)
        for idx, umod in enumerate(uin0):
            umod_one += self.__ds_modulator(umod, ufb0)
            xbit[idx], ufb0 = self.__comp_1bit(umod_one)
            xout_hs[idx] = self.__stream_converter(xbit[idx])

        # --- Downsampling
        xout0 = self.do_decimation_polyphase_order_two(xout_hs)
        xout1 = self.do_decimation_polyphase_order_two(xout0)
        xout2 = self.do_decimation_polyphase_order_two(xout1)
        xout3 = self.do_decimation_polyphase_order_two(xout2)
        xout4 = self.do_decimation_polyphase_order_two(xout3)

        # --- Correction and output
        xout = xout4
        xout -= 2 ** (self._settings.Nadc - 1) if self._settings.type_out == "signed" else 0
        xout = self.digital_clipping(xout)
        return xout

    def adc_deltasigma_order_two(self, uin: np.ndarray) -> np.ndarray:
        """"Using the Delta Sigma Topology as an ADC (1-bit, second order)
        Args:
            uin: Input voltage [V]
        Returns:
            Corresponding digitized data value from Delta-Sigma ADC
        """
        # Resampling the input to sampling frequency of ADC with oversampling
        uin_adc = self.voltage_clipping(uin)
        uin0 = self._do_resample(uin_adc)
        uin0 += self._gen_noise(uin0.size) if self.use_noise else np.zeros(shape=uin0.shape)

        # Running the delta sigma modulator
        xout_hs, xbit = self._generate_dsigma_empty_data(uin0.shape)
        umod_one = self._settings.vcm
        umod_two = self._settings.vcm
        ufb0 = self._settings.vref[1]
        # --- DS Modulator (at high frequency)
        for idx, umod in enumerate(uin0):
            umod_one += self.__ds_modulator(umod, ufb0)
            umod_two += self.__ds_modulator(umod_one, self._settings.vcm)
            xbit[idx], ufb0 = self.__comp_1bit(umod_two)
            xout_hs[idx] = self.__stream_converter(xbit[idx])

        # --- Downsampling
        xout0 = self.do_decimation_polyphase_order_two(xout_hs)
        xout1 = self.do_decimation_polyphase_order_two(xout0)
        xout2 = self.do_decimation_polyphase_order_two(xout1)
        xout3 = self.do_decimation_polyphase_order_two(xout2)
        xout4 = self.do_decimation_polyphase_order_two(xout3)

        # --- Correction and output
        xout = xout4
        xout -= 2 ** (self._settings.Nadc - 1) if self._settings.type_out == "signed" else 0
        xout = self.digital_clipping(xout)
        return xout
