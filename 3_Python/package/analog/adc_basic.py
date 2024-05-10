import dataclasses
import numpy as np
from fractions import Fraction
from scipy.signal import square, resample_poly
from package.data_process.process_noise import noise_awgn


@dataclasses.dataclass
class SettingsADC:
    """"Individuall data class to configure the ADC
    inputs:
    vdd         - Positive supply voltage [V]
    vss         - Negative supply voltage [V]
    dvref       - Half Range of reference voltage [V]
    fs_ana      - Analogue input sampling frequency [Hz]
    fs_dig      - Output sampling rate after decimation [Hz]
    Nadc        - Quantization level of ADC [/]
    osr         - Oversampling ratio of ADC [/]
    type_out    - Output type of digital value {"signed" | "unsigned"}
    """
    vdd:        float
    vss:        float
    dvref:      float
    fs_ana:     float
    fs_dig:     float
    Nadc:       int
    osr:        int
    type_out:   str

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2

    @property
    def fs_adc(self) -> float:
        return self.osr * self.fs_dig

    @property
    def vref(self) -> [float, float]:
        vrefp = self.vcm + self.dvref
        vrefp = vrefp if vrefp < self.vdd else self.vdd
        vrefn = self.vcm - self.dvref
        vrefn = vrefn if vrefn > self.vss else self.vss
        return [vrefp, vrefn]

    @property
    def lsb(self) -> float:
        return (self.vref[0]-self.vref[1]) / (2 ** self.Nadc)


@dataclasses.dataclass
class SettingsNon:
    """Settings for configuring the parasitics/non-linearities of the ADC
    inputs:
    wgndB  - effective power spectral noise [dB/sqrt(Hz)]
    offset - Corner frequency of the flicker (1/f) noise [Hz]
    slope  - Alpha coefficient of the flicker noise []
    """
    wgndB:      float
    offset:     float
    gain_error: float


RecommendedSettingsADC = SettingsADC(
    vdd=0.6, vss=-0.6, dvref=0.1,
    fs_ana=40e3,
    Nadc=12, fs_dig=20e3, osr=1,
    type_out="signed"
)
RecommendedSettingsNon = SettingsNon(
    wgndB=-100,
    offset=1e-6,
    gain_error=0.0
)


class adc_basic:
    """"Basic class for applying an Analogue-Digital-Converter (ADC) on the raw data"""
    def __init__(self, settings_adc: SettingsADC):
        # --- Settings
        self.settings = settings_adc

        # --- Internal characteristic
        self.noise_eff_out = 0.0
        self.__dvrange = self.settings.vref[0] - self.settings.vref[1]
        self.__lsb = self.settings.lsb
        self.__oversampling_ratio = self.settings.osr
        self.__snr_ideal = 10 * np.log10(4) * self.settings.Nadc + 10 * np.log10(3/2)
        self.__digital_border = np.array([0, 2 ** self.settings.Nadc - 1])
        self.__digital_border -= 2 ** (self.settings.Nadc-1) if self.settings.type_out == "signed" else 0
        # --- Resampling stuff
        (self.__p_ratio, self.__q_ratio) = (
            Fraction(self.settings.fs_adc / self.settings.fs_ana)
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
            self.input_snh = uin
        return uout

    def do_snh_stream(self, uin: np.ndarray, f_snh: float) -> np.ndarray:
        """Performing sample-and-hold (S&H) stage for buffering input value"""
        t = np.arange(0, uin.size, 1) / self.settings.fs_ana
        clk_fsh = square(2 * np.pi * t * f_snh, duty=0.5)
        do_snh = np.where(np.diff(clk_fsh) >= 0.5)
        do_snh += 1

        uout = np.zeros(shape=uin.shape)
        for idx, do_snh in enumerate(do_snh):
            uout[idx] = self.__do_snh_sample(uin[idx], do_snh)
        return uout

    def do_resample(self, uin: np.ndarray) -> np.ndarray:
        """Do resampling of input values"""
        uout = uin[0] + resample_poly(uin - uin[0], self.__p_ratio, self.__q_ratio)
        return uout

    def clipping_voltage(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uout = uin
        uout[uin > self.settings.vref[0]] = self.settings.vref[0]
        uout[uin <= self.settings.vref[1]] = self.settings.vref[1]
        return uout

    def clipping_digital(self, xin: np.ndarray) -> np.ndarray:
        """Do digital clipping of quantizied values"""
        xout = xin.astype('int16') if self.settings.type_out == "signed" else xin.astype('uint16')
        xout[xin > self.__digital_border[1]] = self.__digital_border[1]
        xout[xin <= self.__digital_border[0]] = self.__digital_border[0]
        return xout

    def gen_noise(self, size: int) -> np.ndarray:
        """Generate the transient input noise of the amplifier"""
        unoise, self.noise_eff_out = noise_awgn(
            size=size,
            fs=self.settings.fs_ana,
            wgndBW=-self.__snr_ideal
        )
        return unoise

    def adc_ideal(self, uin: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """Using the ideal ADC
        input:
        uin     - Input voltage
        output:
        x_out   - Output digital value
        quant_er - Quantization error
        """
        # Pre-Processing
        uin_adc = self.clipping_voltage(uin)
        uin0 = self.do_resample(uin_adc)
        uin0 += self.gen_noise(uin0.size)
        # ADC conversion
        xout = np.floor((uin0 - self.settings.vcm) / self.__lsb)
        xout = self.clipping_digital(xout)
        uout = self.settings.vref[1] + xout * self.__lsb
        # Calculating quantization error
        uerr = uin0 - uout
        return xout, uout, uerr

    def do_downsample(self, uin: np.ndarray) -> np.ndarray:
        """Performing a simple downsampling of the adc data stream"""
        (p_ratio, q_ratio) = (
            Fraction(self.settings.fs_dig / self.settings.fs_adc)
            .limit_denominator(100)
            .as_integer_ratio()
        )
        uout = uin[0] + resample_poly(uin - uin[0], p_ratio, q_ratio)
        return uout

    def do_cic(self):
        """Performing the CIC filter at the output of oversampled ADC"""
        decimation = 64
        stages = 5
        gain = (decimation * 1) ** stages
        raise NotImplementedError

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
