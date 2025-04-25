from dataclasses import dataclass


@dataclass
class SettingsADC:
    """Individual data class to configure the ADC

    Params:
        vdd         - Positive supply voltage [V]
        vss         - Negative supply voltage [V]
        dvref       - Half Range of reference voltage [V]
        fs_ana      - Analogue input sampling frequency [Hz]
        fs_dig      - Output sampling rate after decimation [Hz]
        Nadc        - Quantization level of ADC [/]
        osr         - Oversampling ratio of ADC [/]
        type_out    - Output type of digital value {"signed": True | "unsigned": False}
    """
    vdd:        float
    vss:        float
    dvref:      float
    fs_ana:     float
    fs_dig:     float
    Nadc:       int
    osr:        int
    is_signed:  bool

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
    def vref_range(self) -> float:
        return self.vref[0] - self.vref[1]

    @property
    def lsb(self) -> float:
        return self.vref_range / (2 ** self.Nadc)


@dataclass
class SettingsNon:
    """Settings for configuring the parasitics/non-linearities of the ADC

    Params:
        use_noise - Boolean for using noise in output
        wgndB  - effective power spectral noise [dB/sqrt(Hz)]
        offset - Corner frequency of the flicker (1/f) noise [Hz]
        slope  - Alpha coefficient of the flicker noise []
    """
    use_noise: bool
    wgndB: float
    offset: float
    gain_error: float


RecommendedSettingsADC = SettingsADC(
    vdd=0.6, vss=-0.6, dvref=0.1,
    fs_ana=40e3,
    Nadc=12, fs_dig=20e3, osr=1,
    is_signed=False
)
RecommendedSettingsNon = SettingsNon(
    use_noise=True,
    wgndB=-100,
    offset=1e-6,
    gain_error=0.0
)
