import dataclasses
import numpy as np
from scipy.signal import lfilter
from package.analog.dev_noise import ProcessNoise, SettingsNoise


@dataclasses.dataclass
class SettingsDLY:
    """Individual data class to configure the delay amplifier
    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        t_dly:      Delay value for shifting the input [s]
        offset:     Offset voltage of the amplifier [V]
        noise_en:   Enable noise on output [True/False]
        noise_edev: Input voltage noise spectral density [V/sqrt(Hz)]
    """
    vdd:    float
    vss:    float
    fs_ana: float
    # Amplifier characteristics
    t_dly:  float
    offset: float
    # Noise properties
    noise_en: bool
    noise_edev: float


RecommendedSettingsDLY = SettingsDLY(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3,
    t_dly=0.1e-3,
    offset=0e-6,
    noise_en=False,
    noise_edev=100e-9
)

RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
)


class DlyAmp(ProcessNoise):
    """Class for emulating an analogue delay amplifier"""
    settings: SettingsDLY
    __print_device = "delay amplifier"

    @property
    def vcm(self) -> float:
        return (self.settings.vdd + self.settings.vss) / 2

    @property
    def num_dly_taps(self) -> int:
        return int(self.settings.fs_ana * self.settings.t_dly)

    def __init__(self, settings_dly: SettingsDLY, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dly.fs_ana)
        # --- Settings
        self.settings = settings_dly

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self.settings.vdd] = self.settings.vdd
        uin[uin < self.settings.vss] = self.settings.vss
        return uin

    def do_simple_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a simple delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros(u_inp.shape) + self.vcm
        uout[self.num_dly_taps:] = u_inp[:-self.num_dly_taps]
        return self.__voltage_clipping(uout)

    def do_recursive_delay(self, u_inp: np.ndarray) -> np.ndarray:
        """Performing a recursive delay stage using taps
        Args:
            u_inp: Applied voltage input [V]
        Returns:
            Corresponding numpy array with shifted input
        """
        uout = np.zeros(u_inp.shape)
        uout[:self.num_dly_taps] = u_inp[-self.num_dly_taps:]
        uout[self.num_dly_taps:] = u_inp[:-self.num_dly_taps]
        return self.__voltage_clipping(uout)

    def do_allpass_first_order(self, uin: np.ndarray, f_b=1.0) -> np.ndarray:
        """Performing a 1st order all-pass filter (IIR) for adding time delay
        Args:
            uin: Input voltage [V]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * f_b / self.settings.fs_ana)
        iir_c0 = (val - 1) / (val + 1)

        b = [iir_c0, 1.0]
        a = [1.0, iir_c0]
        return self.__voltage_clipping(lfilter(b, a, uin))

    def do_allpass_second_order(self, uin: np.ndarray, f_b=1.0, bandwidth=0.5) -> np.ndarray:
        """Performing a 2nd order all-pass filter (IIR) for adding time delay
        Args:
            uin: Input voltage [V]
        Returns:
            Corresponding numpy array with shifted voltage signal
        """
        val = np.tan(np.pi * bandwidth / self.settings.fs_ana)
        iir_c0 = (val - 1) / (val + 1)
        iir_c1 = -np.cos(2 * np.pi * f_b / self.settings.fs_ana)

        b = [-iir_c0, iir_c1*(1-iir_c0), 1.0]
        a = [1.0, iir_c1*(1-iir_c0), -iir_c0]
        return self.__voltage_clipping(lfilter(b, a, uin))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    settings = RecommendedSettingsDLY
    dut = DlyAmp(settings)

    # --- Declaration of input
    t_end = 10e-3
    t0 = np.linspace(0, t_end, num=int(t_end * settings.fs_ana), endpoint=True)
    u_off = 0.0
    u_pp = [0.25, 0.3, 0.1]
    f0 = [1e3, 1.8e3, 2.8e3]
    uinp = np.zeros(t0.shape) + u_off
    for idx, peak_val in enumerate(u_pp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * f0[idx])
    uinn = dut.vcm

    uout0 = dut.do_simple_delay(uinp)
    uout1 = dut.do_recursive_delay(uinp)
    uout2 = dut.do_allpass_first_order(uinp, 1e3)
    uout3 = dut.do_allpass_second_order(uinp, 1e3, 100)

    # --- Plotting
    plt.figure()
    plt.plot(t0, uinp, 'r', label="Input")
    plt.plot(t0, uout0, 'k', label="Out (simple)")
    plt.plot(t0, uout1, 'm', label="Out (recursive)")
    plt.plot(t0, uout2, 'y', label="Out (all-pass)")

    plt.xlabel('Time t / s')
    plt.ylabel('Voltage U_x / V')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
