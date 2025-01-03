import numpy as np
from fractions import Fraction
from scipy.signal import square, resample_poly
from package.analog.adc.adc_settings import SettingsADC
from package.analog.dev_noise import noise_awgn


class BasicADC:
    """"Basic class for applying an Analogue-Digital-Converter (ADC) on the raw data"""
    def __init__(self, settings_adc: SettingsADC):
        # --- Settings
        self._settings = settings_adc

        # --- Internal characteristic
        self.noise_eff_out = 0.0
        self.__dvrange = self._settings.vref[0] - self._settings.vref[1]
        self.__lsb = self._settings.lsb
        self.__oversampling_ratio = self._settings.osr
        self.__snr_ideal = 10 * np.log10(4) * self._settings.Nadc + 10 * np.log10(3 / 2)
        self.__digital_border = np.array([0, 2 ** self._settings.Nadc - 1])
        self.__digital_border -= 2 ** (self._settings.Nadc - 1) if self._settings.type_out == "signed" else 0
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

    def _clipping_voltage(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uout = np.zeros(uin.shape) + uin
        if uin.size == 1:
            uout = uout if not uin > self._settings.vdd else self._settings.vdd
            uout = uout if not uin < self._settings.vss else self._settings.vss
        else:
            xpos = np.argwhere(uin > self._settings.vdd)
            xneg = np.argwhere(uin < self._settings.vss)
            uout[xpos] = self._settings.vdd
            uout[xneg] = self._settings.vss
        return uout

    def _clipping_digital(self, xin: np.ndarray) -> np.ndarray:
        """Do digital clipping of quantizied values"""
        xout = xin.astype('int16') if self._settings.type_out == "signed" else xin.astype('uint16')
        xout[xin > self.__digital_border[1]] = self.__digital_border[1]
        xout[xin <= self.__digital_border[0]] = self.__digital_border[0]
        return xout

    def _gen_noise(self, size: int) -> np.ndarray:
        """Generate the transient input noise of the amplifier"""
        unoise = noise_awgn(
            size=size,
            fs=self._settings.fs_ana,
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
        uin_adc = self._clipping_voltage(uin)
        uin0 = self._do_resample(uin_adc)
        uin0 += self._gen_noise(uin0.size)
        # ADC conversion
        xout = np.floor((uin0 - self._settings.vcm) / self.__lsb)
        xout = self._clipping_digital(xout)
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

    def do_decimation_polyphase_order_one(self, uin: np.ndarray) -> np.ndarray:
        """Performing first order Non-Recursive Polyphase Decimation on input"""
        last_sample_hs = 0
        uout = []
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_hs)
            last_sample_hs = val

        uout = np.array(uout)
        return uout

    def do_decimation_polyphase_order_two(self, uin: np.ndarray) -> np.ndarray:
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
