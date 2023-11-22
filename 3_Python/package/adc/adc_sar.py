import numpy as np
from package.adc.adc_basic import ADC_Basic, SettingsADC, RecommendedSettingsNon

class ADC_SAR(ADC_Basic):
    """"Class for applying a Sukzessive Approximation (SAR) Analogue-Digital-Converter (ADC) on the raw data"""
    def __init__(self, setting: SettingsADC):
        super().__init__(setting, RecommendedSettingsNon)
        self.use_noise = True
        # --- Transfer function
        self.__dv = self.settings.vref[0] - self.settings.vref[1]
        self.__partition_digital = 2 ** np.arange(0, self.settings.Nadc)
        self.__partition_voltage = (self.__partition_digital / 2 ** self.settings.Nadc) * self.__dv
        self.__type_offset = [2 ** (self.settings.Nadc-1) if self.settings.type_out == "signed" else 0]
        # --- Internal signals for noise shaping
        self.alpha_int = [1, 0.5]
        self.__stage_one_dly = self.settings.vcm
        self.__stage_two_dly = self.settings.vcm

    def __adc_sar_sample(self, uin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Running the SAR on input data"""
        # --- Bitmask generation
        BitMask = np.zeros(shape=(self.settings.Nadc,), dtype=int)
        BitMask[-1] = 1
        # --- Run SAR code
        for idx in range(0, self.settings.Nadc):
            uref = self.settings.vref[1] + np.sum(BitMask * self.__partition_voltage)
            BitMask[self.settings.Nadc - 1 - idx] = np.heaviside(uin - uref, 1)
            if not idx == self.settings.Nadc - 1:
                BitMask[self.settings.Nadc-2-idx] = 1

        uout = self.settings.vref[1] + np.sum(BitMask * self.__partition_voltage)
        xout = (np.sum(BitMask * self.__partition_digital) - self.__type_offset).astype(int)
        return uout, xout

    def adc_sar(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the SAR Topology as an ADC
        input:
        uin     - Input voltage
        output:
        x_out   - Output digital value
        quant_er - Quantization error
        """
        # Resampling of input
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)
        # Running SAR code
        uout = np.zeros(shape=uin0.shape)
        xout = np.zeros(shape=uin0.shape)
        uerr = np.zeros(shape=uin0.shape)
        for idx, umod in enumerate(uin0):
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self.settings.lsb)
            uerr[idx] = umod - uout[idx]
        return xout, uout, uerr

    def adc_sar_ns_delay(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (Delay of last sample)
        input:
        uin     - Input voltage
        output:
        x_out   - Output digital value
        quant_er - Quantization error
        """
        # Resampling of input
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)
        # Running SAR code
        uout = np.zeros(shape=uin0.shape)
        xout = np.zeros(shape=uin0.shape)
        uerr = np.zeros(shape=uin0.shape)
        for idx, din in enumerate(uin0):
            umod = din + self.__stage_one_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self.settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Noise shaping post-processing
            self.__stage_one_dly = uerr[idx]
        return xout, uout, uerr

    def adc_sar_ns_order_one(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (First order with integration)
        input:
        uin     - Input voltage
        output:
        x_out   - Output digital value
        quant_er - Quantization error
        """
        # Resampling of input
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(shape=uin0.shape)
        # Running SAR code
        uout = np.zeros(shape=uin0.shape)
        xout = np.zeros(shape=uin0.shape)
        uerr = np.zeros(shape=uin0.shape)
        for idx, din in enumerate(uin0):
            # --- SAR processing
            umod = din + self.__stage_one_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self.settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Post-processing: Noise shaping
            self.__stage_one_dly += self.alpha_int[0] * uerr[idx]
        return xout, uout, uerr

    def adc_sar_ns_order_two(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Running the Noise Shaping SAR Topology (Second order with integration)
        input:
        uin         - Input voltage
        output:
        x_out       - Output digital value
        quant_er    - Quantization error
        """
        # Resampling of input
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        unoise = self.gen_noise(uin0.size) if self.use_noise == True else np.zeros(
            shape=uin0.shape)
        # Running SAR code
        uout = np.zeros(shape=uin0.shape)
        xout = np.zeros(shape=uin0.shape)
        uerr = np.zeros(shape=uin0.shape)
        for idx, din in enumerate(uin0):
            umod = din + self.__stage_two_dly
            calc_out = self.__adc_sar_sample(umod)
            uout[idx] = calc_out[0] + unoise[idx]
            xout[idx] = calc_out[1] + np.floor(unoise[idx] / self.settings.lsb)
            uerr[idx] = din - uout[idx]
            # --- Noise shaping post-processing
            self.__stage_one_dly += self.alpha_int[0] * uerr[idx]
            self.__stage_two_dly += self.alpha_int[1] * self.__stage_one_dly
        return xout, uout, uerr

# ------------ TEST ROUTINE -------------
from package.data.process_noise import noise_real, do_fft
import matplotlib.pyplot as plt
if __name__ == "__main__":
    set_adc = SettingsADC(
        vdd=0.6, vss=-0.6,
        fs_ana=200e3, fs_dig=20e3, osr=10,
        dvref=0.1, Nadc=12,
        type_out="signed"
    )
    adc0 = ADC_SAR(set_adc)

    t_end = 1
    tA = np.arange(0, t_end, 1/set_adc.fs_ana)
    tD = np.arange(0, t_end, 1/set_adc.fs_dig)
    # --- Input signal
    upp = 0.8 * set_adc.dvref
    fsine = 100
    uin = upp * np.sin(2 * np.pi * tA * fsine)
    uin += noise_real(tA.size, tA.size, -120, 1, 0.6)[0]
    # --- ADC output
    uadc_hs = adc0.adc_sar_ns_order_one(uin)[0]
    uadc = adc0.do_downsample(uadc_hs)
    freq, Yadc = do_fft(uadc_hs, set_adc.fs_adc)

    # --- Plotting results
    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313)

    vscale = 1e3
    ax1.plot(tA, vscale * uin)
    ax1.set_ylabel('U_in [mV]')
    ax1.set_xlim([100e-3, 150e-3])

    ax2.plot(tD, uadc)
    ax2.set_ylabel('X_adc []')
    ax2.set_xlabel('Time t [s]')

    ax3.semilogx(freq, 20 * np.log10(Yadc))
    ax3.set_ylabel('X_adc []')
    ax3.set_xlabel('Frequency f [Hz]')

    plt.tight_layout()
    plt.show(block=True)
