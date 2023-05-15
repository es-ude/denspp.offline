import numpy as np
from src.adc.adc_basic import ADC_Basic, SettingsADC, SettingsNon, RecommendedSettingsADC, RecommendedSettingsNon

class ADC_DeltaSigma(ADC_Basic):
    """Class for using Continuous Time Delta Sigma ADC"""
    def __init__(self, setting: SettingsADC):
        super().__init__(setting, RecommendedSettingsNon())
        # --- Internal variables
        self.use_noise = False
        self.__dac_order = 2
        self.__dac_dvrange = self.settings.vref[0] - self.settings.vref[1]
        self.__dac_lsb = self.__dac_dvrange / 2 ** self.__dac_order
        self.__partition_digital = np.arange(0, 2 ** self.__dac_order, 1) / 2 ** self.__dac_order
        self.__partition_voltage = self.settings.vref[1] + self.__partition_digital * self.__dac_dvrange + self.__dac_lsb / 2

        # --- Variables for post-processing (noise-shaping)
        self.__stage_one_dly = self.settings.vcm
        self.__stage_two_dly = self.settings.vcm
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
        du = du if not du > self.settings.vdd else self.settings.vdd
        du = du if not du < self.settings.vss else self.settings.vss
        # Output
        return du

    def __stream_converter(self, xin: int) -> np.ndarray:
        """Performing the stream conversion"""
        xout = (1 + np.sum((-1) ** (1 - xin))) / 2
        return xout

    def __comp_1bit(self, uin: float) -> [np.ndarray, np.ndarray]:
        """1-bit DAC for DS modulation"""
        xout = np.heaviside(uin - self.settings.vcm, 1)
        ufb = self.settings.vref[0] if xout == 1 else self.settings.vref[1]
        return xout, ufb

    def __comp_Nbit(self, uin: float) -> [np.ndarray, np.ndarray]:
        """N-bit DAC for DS modulation"""
        input = uin * np.ones(shape=self.__partition_voltage.shape)
        result = np.heaviside(input - self.__partition_voltage, 1)
        xout = np.sum(result)
        ufb = self.settings.vref[1] + xout * self.__dac_lsb

        return xout, ufb

    def adc_deltasigma_order_one(self, uin: np.ndarray) -> np.ndarray:
        """"Using the Delta Sigma Topology as an ADC (1-bit, first order)"""
        # Resampling the input to sampling frequency of ADC with oversampling
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)
        # Running the delta sigma modulator
        xout_hs = np.zeros(shape=uin0.shape)
        xbit = np.zeros(shape=uin0.shape)
        umod_one = self.settings.vcm
        ufb0 = self.settings.vref[1]
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
        xout -= 2 ** (self.settings.Nadc - 1) if self.settings.type_out == "signed" else 0
        xout = self.clipping_digital(xout)
        return xout

    def adc_deltasigma_order_two(self, uin: np.ndarray) -> np.ndarray:
        """"Using the Delta Sigma Topology as an ADC (1-bit, first order)"""
        # Resampling the input to sampling frequency of ADC with oversampling
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)
        # Running the delta sigma modulator
        xout_hs = np.zeros(shape=uin0.shape)
        xbit = np.zeros(shape=uin0.shape)
        umod_one = self.settings.vcm
        umod_two = self.settings.vcm
        ufb0 = self.settings.vref[1]
        # --- DS Modulator (at high frequency)
        for idx, umod in enumerate(uin0):
            umod_one += self.__ds_modulator(umod, ufb0)
            umod_two += self.__ds_modulator(umod_one, self.settings.vcm)
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
        xout -= 2 ** (self.settings.Nadc - 1) if self.settings.type_out == "signed" else 0
        xout = self.clipping_digital(xout)
        return xout

# ----- TEST ROUTINE ------------
import matplotlib.pyplot as plt
from src.processing_noise import noise_real, do_fft
if __name__ == "__main__":
    set_adc = SettingsADC(
        vdd=0.6, vss=-0.6,
        fs_ana=200e3, fs_dig=20e3, osr=32,
        dvref=0.1, Nadc=10,
        type_out="signed"
    )
    adc0 = ADC_DeltaSigma(set_adc)

    t_end = 1
    tA = np.arange(0, t_end, 1 / set_adc.fs_ana)
    tD = np.arange(0, t_end, 1 / set_adc.fs_dig)
    # --- Input signal
    upp = 0.8 * set_adc.dvref
    fsine = 100
    uin = upp * np.sin(2 * np.pi * tA * fsine)
    uin += noise_real(tA.size, tA.size, -120, 1, 0.6)[0]
    # --- ADC output
    xout = adc0.adc_deltasigma_order_two(uin)
    freq, Yadc = do_fft(xout, set_adc.fs_dig)

    # --- Plotting results
    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313)

    vscale = 1e3
    ax1.plot(tA, vscale * uin)
    ax1.set_ylabel('U_in [mV]')
    ax1.set_xlim([100e-3, 150e-3])

    ax2.plot(tD, xout)
    ax2.set_ylabel('X_adc []')
    ax2.set_xlabel('Time t [s]')

    ax3.semilogx(freq, 20 * np.log10(Yadc))
    ax3.set_ylabel('X_adc []')
    ax3.set_xlabel('Frequency f [Hz]')

    plt.tight_layout()
    plt.show(block=True)
